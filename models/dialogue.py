import os
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from torchtext.data import LabelField, Field, Example, Dataset, BucketIterator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from conlleval import evaluate as conll_lines

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


"""
!pip install torch numpy sklearn conlleval
!mkdir -p data
!git clone https://github.com/MiuLab/SlotGated-SLU.git
!mv SlotGated-SLU data
"""  # noqa


def read_single(path):
    with open(os.path.join(path, 'seq.in')) as fwords, \
            open(os.path.join(path, 'seq.out')) as ftags, \
            open(os.path.join(path, 'label')) as fintents:

        df = pd.DataFrame({
            "toekns": [w.strip().split() for w in fwords],
            "tags": [t.strip().split() for t in ftags],
            "intent": [i.strip() for i in fintents],
        })
    return df


def data():
    return (
        read_single("data/SlotGated-SLU/data/atis/train/"),
        read_single("data/SlotGated-SLU/data/atis/test/"),
        read_single("data/SlotGated-SLU/data/atis/valid/"),
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for (name, field) in self.fields:
            field.build_vocab(dataset)
        return self

    def transform(self, X, y=None):
        cols = [X[col].apply(field.preprocess) for col, field in self.fields]
        examples = [Example.fromlist(c, self.fields) for c in zip(*cols)]
        return Dataset(examples, self.fields)


def build_preprocessor():
    fields = [
        ('tokens', Field()),
        ('tags', Field(unk_token=None)),
        ('intent', LabelField()),
    ]
    return TextPreprocessor(fields)


class IntentClassifierModel(torch.nn.Module):
    def __init__(self, vocab_size, output_count, emb_dim=64,
                 lstm_hidden_dim=128, num_layers=1):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim, num_layers=num_layers)
        self._out = torch.nn.Linear(lstm_hidden_dim, output_count)

    def forward(self, inputs):
        rnn_output = self._rnn(self._emb(inputs))[0]
        return self._out(rnn_output[-1])


class ModelTrainer():
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def on_epoch_begin(self, is_train, name, batches_count):
        """
        Initialize metrics
        """
        self.epoch_loss = 0
        self.correct_count = 0
        self.total_count = 0
        self.is_train = is_train
        self.name = name
        self.batches_count = batches_count
        self.model.train(is_train)

    def on_epoch_end(self):
        msg = '{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'
        return msg.format(
            self.name,
            self.epoch_loss / self.batches_count,
            self.correct_count / self.total_count
        )

    def _loss(self, batch):
        logits = self.model(batch.tokens)
        pred = logits.argmax(-1)

        self.correct_count += (pred == batch.intent).float().sum()
        self.total_count += len(batch.intent)

        return self.criterion(logits, batch.intent)

    def on_batch(self, batch):
        loss = self._loss(batch)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
        self.epoch_loss += loss.item()

        return '{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
            self.name, loss.item(), self.correct_count / self.total_count
        )

    def epoch(self, data_iter, pad_idx, is_train, name=None):
        self.on_epoch_begin(is_train, name, batches_count=len(data_iter))

        with torch.autograd.set_grad_enabled(is_train):
            with tqdm(total=self.batches_count) as progress_bar:
                for i, batch in enumerate(data_iter):
                    batch_progress = self.on_batch(batch)

                    progress_bar.update()
                    progress_bar.set_description(batch_progress)

                epoch_progress = self.on_epoch_end()
                progress_bar.set_description(epoch_progress)
                progress_bar.refresh()


class TaggerModel(torch.nn.Module):
    def __init__(self, vocab_size, output_count,
                 emb_dim=64, lstm_hidden_dim=128,
                 num_layers=1, bidirectional=True):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional)

        self._out = torch.nn.Linear(
            (1 + bidirectional) * lstm_hidden_dim * num_layers, output_count)

    def forward(self, inputs):
        embs = self._emb(inputs)
        outputs, _ = self._rnn(embs, None)
        return self._out(outputs)


class TaggerTrainer(ModelTrainer):
    def _loss(self, batch):
        target = batch.tags
        logits = self.model(batch.tokens)

        mask = (target != self.pad_idx).float()
        pred = logits.argmax(-1)

        self.correct_count += ((pred == batch.tags).float() * mask).sum()
        self.total_count += mask.sum()

        return self.criterion(
            logits.view(-1, logits.shape[-1]), target.view(-1))


class UnifiedClassifier(BaseEstimator, TransformerMixin):
    _modeltypes = {
        "intent": IntentClassifierModel,
        "tagger": TaggerModel,
    }

    _trainertypes = {
        "intent": ModelTrainer,
        "tagger": TaggerTrainer,
    }

    _targets = {
        "intent": "intent",
        "tagger": "tags"
    }

    def __init__(self, modelname="intent", batch_size=32,
                 epochs_count=30, model=None):
        self.model = None
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.trainer = None
        self._target = self._targets[modelname]
        self._modeltype = self._modeltypes[modelname]
        self._trainertype = self._trainertypes[modelname]

    def _init_trainer(self, X, y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.trainer is None:
            self.model = self._modeltype(
                vocab_size=len(X.fields["tokens"].vocab),
                output_count=len(X.fields[self._target].vocab)).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(self.model.parameters())
            self.trainer = self._trainertype(self.model, criterion, optimizer)
            self.trainer.pad_idx = X.fields["tokens"].vocab.stoi["pad"]

    def fit(self, X, y=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._init_trainer(X, y)
        train_dataset, test_dataset = X.split(split_ratio=0.7)

        train, val = BucketIterator.splits(
            (train_dataset, test_dataset),
            batch_sizes=(self.batch_size, self.batch_size * 4),
            shuffle=True,
            device=device,
            sort=False,
        )

        pad_idx = X.fields["tokens"].vocab.stoi["<pad>"]
        for epoch in range(self.epochs_count):
            name = '[{} / {}] Train'.format(epoch + 1, self.epochs_count)
            self.trainer.epoch(train, pad_idx, is_train=True, name=name)
            name = '[{} / {}] Val'.format(epoch + 1, self.epochs_count)
            self.trainer.epoch(val, pad_idx, is_train=False, name=name)
        return self

    def predict(self, X):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        x_iter = BucketIterator(
            X, batch_size=self.batch_size * 4, device=device)

        output = []
        pi = X.fields["tokens"].vocab.stoi["<pad>"]
        vocab = X.fields[self._target].vocab.itos
        with torch.no_grad():
            for batch in x_iter:
                pred = self.model(batch.tokens).argmax(-1)
                if self._target == "intent":
                    labels = [X.fields[self._target].vocab.itos[i]
                              for i in pred]
                    output.append(labels)
                    continue

                mask = batch.tokens != pi
                for seq, m in zip(pred.T, mask.T):
                    output.append([vocab[i] for i in seq[m]])
        return output


class SharedModel(torch.nn.Module):
    def __init__(self, vocab_size, intents_count, tags_count,
                 emb_dim=64, lstm_hidden_dim=128, num_layers=1):
        super().__init__()

        self._intents_embs = torch.nn.Embedding(vocab_size, emb_dim)
        self._tags_embs = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(2 * emb_dim, lstm_hidden_dim,
                                  num_layers=num_layers, bidirectional=True)
        self._intents_out = torch.nn.Linear(
            2 * lstm_hidden_dim * num_layers, intents_count)
        self._tags_out = torch.nn.Linear(
            2 * lstm_hidden_dim * num_layers, tags_count)

    def forward(self, inputs):
        intents_embs = self._intents_embs(inputs)
        tags_embs = self._tags_embs(inputs)

        embs = torch.cat((intents_embs, tags_embs), -1)

        output, (hidden, _) = self._rnn(embs, None)
        intents = torch.cat((hidden[0], hidden[1]), -1)

        intents = self._intents_out(intents).squeeze(0)
        tags = self._tags_out(output)

        return intents, tags


class SharedTrainer():
    _msg = (
        "{:>5s} Loss = {:.5f},"
        " Intents Accuracy = {:.2%},"
        " Tags Accuracy = {:.2%}"
    )

    def __init__(self, model, intents_criterion,
                 tags_criterion, optimizer, pad_idx):
        self.model = model
        self.intents_criterion = intents_criterion
        self.tags_criterion = tags_criterion
        self.optimizer = optimizer
        self.pad_idx = pad_idx

    def on_epoch_begin(self, is_train, name, batches_count):
        """
        Initializes metrics
        """
        self.epoch_loss = 0
        self.intents_correct_count, self.intents_total_count = 0, 0
        self.tags_correct_count, self.tags_total_count = 0, 0
        self.is_train = is_train
        self.name = name
        self.batches_count = batches_count

        self.model.train(is_train)

    def on_epoch_end(self):
        """
        Outputs final metrics
        """
        return self._msg.format(
            self.name, self.epoch_loss / self.batches_count,
            self.intents_correct_count / self.intents_total_count,
            self.tags_correct_count / self.tags_total_count
        )

    def on_batch(self, batch):
        """
        Performs forward and (if is_train) backward pass with optimization,
        updates metrics
        """
        intents, tags = self.model(batch.tokens)
        intents_pred = intents.argmax(-1)
        tags_pred = tags.argmax(-1)

        mask = (batch.tags != self.pad_idx).float()

        self.intents_correct_count += (
            intents_pred == batch.intent).float().sum()
        self.intents_total_count += len(batch.intent)
        self.tags_correct_count += (
            (tags_pred == batch.tags).float() * mask).sum()
        self.tags_total_count += mask.sum()

        intents_loss = self.intents_criterion(intents, batch.intent)
        tags_loss = self.tags_criterion(
            tags.view(-1, tags.shape[-1]), batch.tags.view(-1))
        loss = intents_loss + tags_loss

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
        self.epoch_loss += loss.item()

        return self._msg.format(
            self.name, loss.item(),
            self.intents_correct_count / self.intents_total_count,
            self.tags_correct_count / self.tags_total_count
        )


def conll_score(y_true, y_pred, metrics=("f1", "prec", "rec"), **kwargs):
    lines = [f"dummy XXX {t} {p}" for pair in zip(y_true, y_pred)
             for t, p in zip(*pair)]
    res = conll_lines(lines)
    return [res["overall"]["tags"]["evals"][m] for m in metrics]


def build_model(**args):
    model = make_pipeline(
        build_preprocessor(),
        UnifiedClassifier(**args),
    )
    return model


def main():
    train, test, valid = data()
    print(train.head())


if __name__ == '__main__':
    main()
