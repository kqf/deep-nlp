import torch
import random
import skorch
import numpy as np
import itertools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from torchtext.data import Field, Example, Dataset, BucketIterator
from conlleval import evaluate as conll_lines


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def read_dataset(path):
    data = []
    with open(path) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line and words:
                data.append((words, tags))
                words, tags = [], []
                continue
            word, pos_tag, synt_tag, ner_tag = line.split()
            words.append(word)
            tags.append(ner_tag)
        if words:
            data.append((words, tags))
    return data[1:]


def data(dataset="train"):
    return read_dataset(f"data/conll_2003/{dataset}.txt")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            field.build_vocab(dataset)
        return self

    def transform(self, X, y=None):
        examples = [Example.fromlist(f, self.fields) for f in X]
        dataset = Dataset(examples, self.fields)
        return dataset


def build_preprocessor():
    fields = [
        ('tokens', Field(unk_token=None, batch_first=True)),
        ('tags', Field(unk_token=None, batch_first=True, is_target=True)),
    ]
    return TextPreprocessor(fields)


class BaselineTagger(torch.nn.Module):
    def __init__(self, embeddings, tags_count=2, rnn_dim=256, num_layers=1):
        super().__init__()
        self._out = torch.nn.Linear(rnn_dim, tags_count)
        if embeddings is None:
            return

        self._emb = torch.nn.Embedding.from_pretrained(embeddings)
        emb_dim = embeddings.shape[1]
        self._rnn = torch.nn.LSTM(
            emb_dim, rnn_dim, num_layers=num_layers, batch_first=True)

    def forward(self, inputs):
        emb = self._emb(inputs)
        hid, _ = self._rnn(emb)
        return self._out(hid)


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        svocab = X.fields["tokens"].vocab
        tvocab = X.fields["tags"].vocab
        # TODO: Fix me later
        embeddings = torch.rand(len(svocab), 100)
        net.set_params(module__embeddings=embeddings)
        net.set_params(module__tags_count=len(tvocab))
        net.set_params(criterion__ignore_index=tvocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TaggerNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, y_true.view(-1))

    def predict(self, X):
        probs = self.predict_proba(X)
        label_ids = probs.argmax(-1)
        return np.take(X.fields["tags"].vocab.itos, label_ids)

    def score(self, X, y):
        preds = self.predict(X)
        trimmed = [p[:len(t)] for p, t in zip(preds, y)]
        y_pred = list(itertools.chain(*trimmed))
        y_true = list(itertools.chain(*y))
        return conll_score(y_true, y_pred)


def conll_score(y_true, y_pred, metrics="f1", **kwargs):
    lines = [f"dummy XXX {t} {p}" for pair in zip(y_true, y_pred)
             for t, p in zip(*pair)]
    result = conll_lines(lines)["overall"]["tags"]["evals"]

    if isinstance(metrics, str):
        return result[metrics]

    return [result[m] for m in metrics]


def build_baseline():
    model = TaggerNet(
        module=BaselineTagger,
        module__embeddings=None,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=4,
        batch_size=64,
        iterator_train=BucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=BucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            DynamicVariablesSetter(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    df = data()
    print(df[:3])
    build_preprocessor().fit_transform(df)


if __name__ == '__main__':
    main()
