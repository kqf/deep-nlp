import torch
import skorch
import random
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.data import BucketIterator


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def data():
    fname = "data/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    return pd.read_csv(fname)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        return Dataset(examples, self.fields)


class VanilaRNN(torch.nn.Module):
    def __init__(self, vocab_size, n_sentiments, emb_dim=100, hid_dim=256):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.RNN(emb_dim, hid_dim)
        self._out = torch.nn.Linear(hid_dim, n_sentiments)

    def forward(self, text):
        # text[seq_len, batch size] -> [seq_len, batch size, emb dim] -> RNN
        output, hidden = self._rnn(self._emb(text))
        # output = [seq_len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        return self._out(hidden.squeeze(0))


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, n_sentiments,
                 emb_dim=100,
                 lstm_hidden_dim=256,
                 lstm_layers_count=1, bidirectional=False):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(
            emb_dim, lstm_hidden_dim,
            lstm_layers_count, bidirectional=bidirectional)

        hidden = 2 * lstm_hidden_dim if bidirectional else lstm_hidden_dim
        self._out = torch.nn.Linear(hidden, n_sentiments)

    def forward(self, inputs):
        output, _ = self._rnn(self._emb(inputs))
        hidden = output[-1, :, :]
        if self._rnn.bidirectional:
            hidden = torch.cat([hidden, output[-2, :, :]], dim=1)
        return self._out(output[-1, :, :])


def build_preprocessor(packed=False):
    fields = [
        ("review", Field(include_lengths=packed)),
        ("sentiment", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


def build_model(module=VanilaRNN):
    model = skorch.NeuralNet(
        module=module,
        module__vocab_size=10,  # Dummy dimension
        module__n_sentiments=2,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=BucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=BucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    df = data()
    train, test = train_test_split(df, test_size=0.5)

    print(train.head())
    print(train.info())


if __name__ == '__main__':
    main()
