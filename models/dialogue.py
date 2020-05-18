import os
import torch
import random
import numpy as np
import pandas as pd

import torch
from torchtext.data import LabelField, Field, Example, Dataset
from sklearn.base import BaseEstimator, TransformerMixin
# from torchtext.data import Dataset, BucketIterator

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


"""
!mkdir -p data
!git clone https://github.com/MiuLab/SlotGated-SLU.git
!mv SlotGated-SLU data
"""  # noqa


def read_single(path):
    with open(os.path.join(path, 'seq.in')) as fwords, \
            open(os.path.join(path, 'seq.out')) as ftags, \
            open(os.path.join(path, 'label')) as fintents:

        df = pd.DataFrame({
            "words": [w.strip().split() for w in fwords],
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
        ('words', Field()),
        ('tags', Field(unk_token=None)),
        ('intent', LabelField()),
    ]
    return TextPreprocessor(fields)


class IntentClassifierModel(torch.nn.Module):
    def __init__(self, vocab_size, intents_count, emb_dim=64,
                 lstm_hidden_dim=128, num_layers=1):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim, num_layers=num_layers)
        self._out = torch.nn.Linear(lstm_hidden_dim, intents_count)

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

    def on_batch(self, batch):
        logits = self.model(batch.tokens)
        pred = logits.argmax(-1)

        self.correct_count += (pred == batch.intent).float().sum()
        self.total_count += len(batch.intent)

        loss = self.criterion(logits, batch.intent)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
        self.epoch_loss += loss.item()

        return '{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
            self.name, loss.item(), self.correct_count / self.total_count
        )


def main():
    train, test, valid = data()
    print(train.head())


if __name__ == '__main__':
    main()
