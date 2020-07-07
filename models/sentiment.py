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


def context(inputs, bidirectional=False):
    if bidirectional:
        return torch.cat([inputs[-1, :, :], inputs[-2, :, :]], dim=-1)
    return inputs[-1, :, :]


class VanilaRNN(torch.nn.Module):
    def __init__(self, vocab_size, n_sentiments, emb_dim=100, hid_dim=256,
                 padding_idx=0,
                 bidirectional=False):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self._rnn = torch.nn.RNN(emb_dim, hid_dim,
                                 bidirectional=bidirectional)
        hid_dim = 2 * hid_dim if bidirectional else hid_dim
        self._out = torch.nn.Linear(hid_dim, n_sentiments)

    def forward(self, text):
        # text[seq_len, batch_size] -> [seq_len, batch_size, emb_dim] -> RNN
        _, hidden = self._rnn(self._emb(text))
        # output = [seq_len, batch_size, hid dim]
        return self._out(context(hidden, self._rnn.bidirectional))


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, n_sentiments, emb_dim=100, hid_dim=256,
                 lstm_layers_count=1, padding_idx=0, bidirectional=False):
        super().__init__()

        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self._rnn = torch.nn.LSTM(
            emb_dim, hid_dim,
            lstm_layers_count, bidirectional=bidirectional)

        hid_dim = 2 * hid_dim if bidirectional else hid_dim
        self._out = torch.nn.Linear(hid_dim, n_sentiments)

    def forward(self, inputs):
        _, (hidden, _) = self._rnn(self._emb(inputs))
        return self._out(context(hidden, self._rnn.bidirectional))


class PackedLSTM(LSTM):
    def forward(self, inputs):
        text, lengths = inputs
        # Transpose to handle batch first required by skorch
        embs = self._emb(text.T)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embs, lengths)
        output, (hidden, _) = self._rnn(packed)

        # for unpacking:
        # output, out = nn.utils.rnn.pad_packed_sequence(output)
        return self._out(context(hidden))


class FastText(torch.nn.Module):
    def __init__(self,
                 vocab_size, n_sentiments,
                 emb_dim=100, padding_idx=0,
                 bidirectional="dummy"):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self._out = torch.nn.Linear(emb_dim, n_sentiments)

    def forward(self, text):
        # embe = [seq_len, batch_size, emb_dim]
        emb = self._emb(text)

        # emb = [batch_size, seq_len, emb_dim]
        emb = emb.permute(1, 0, 2)

        # pooled = [batch_size, 1, emb_dim]
        pooled = torch.nn.functional.avg_pool2d(emb, (emb.shape[1], 1))

        # pooled -> [batch_size, emb_dim]
        return self._out(pooled.squeeze(1))


class CNN(torch.nn.Module):
    def __init__(
        self,
        vocab_size, n_sentiments,
        emb_dim=100, n_filters=100, filter_sizes=None,
        dropout=0.5, padding_idx=0,
        bidirectional="dummy",
        conv=torch.nn.Conv2d
    ):
        super().__init__()
        filter_sizes = filter_sizes or [1, 2, 3]
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self._convs = torch.nn.ModuleList([
            conv(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(fs, emb_dim)
            )
            for fs in filter_sizes
        ])
        self._out = torch.nn.Linear(
            len(filter_sizes) * n_filters, n_sentiments)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, text):

        # [seq_len, batch_size] -> [seq_len, batch_size, emb_dim]
        emb = self._emb(text)

        # emb = [batch_size, seq_len, emb_dim]
        emb = emb.permute(1, 0, 2)

        # emb = [batch_size, 1, seq_len, emb_dim]
        emb = emb.unsqueeze(1)

        # conved_i = [batch_size, n_filters, seq_len - filter_sizes[n] + 1]
        conved = [
            torch.nn.functional.relu(conv(emb)).squeeze(3)
            for conv in self._convs]

        # pooled_i = [batch_size, n_filters]
        pooled = [
            torch.nn.functional.max_pool1d(c, c.shape[2]).squeeze(2)
            for c in conved]

        # cat = [batch_size, n_filters * len(filter_sizes)]
        cat = self._dropout(torch.cat(pooled, dim=1))
        return self._out(cat)


def ngrams(x, n=2):
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def build_preprocessor(packed=False, preprocessing=None):
    text_field = Field(
        include_lengths=packed,
        batch_first=packed,
        preprocessing=preprocessing,
    )

    fields = [
        ("review", text_field),
        ("sentiment", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


class DynamicParSertter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["review"].vocab
        svocab = X.fields["sentiment"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__n_sentiments=len(svocab))
        net.set_params(module__padding_idx=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(module=VanilaRNN, packed=False, bidirectional=False):
    model = skorch.NeuralNet(
        module=module,
        module__vocab_size=10,  # Dummy dimension
        module__n_sentiments=2,
        module__bidirectional=bidirectional,
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
            DynamicParSertter(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(packed),
        model,
    )
    return full


def main():
    df = data()
    train, test = train_test_split(df, test_size=0.5)

    print(train.head())
    print(train.info())

    vrnn = build_model(module=VanilaRNN, bidirectional=True)
    vrnn.fit(train)
    y_pred = vrnn.predict(test)

    print(y_pred)


if __name__ == '__main__':
    main()
