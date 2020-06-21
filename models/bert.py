import torch
import skorch
import numpy as np
import pandas as pd

from torchtext.data import Field, LabelField, Dataset, Example
from torchtext.data import BucketIterator

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


def data(dataset="train"):
    df = pd.read_csv(f"data/{dataset}.csv.zip")
    df.replace(np.nan, '', regex=True, inplace=True)
    return df


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=0):
        self.fields = fields
        self.min_freq = min_freq or {}

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


def build_preprocessor():
    text_field = Field(
        batch_first=True
    )
    fields = [
        ('question1', text_field),
        ('question2', text_field),
        ('is_duplicate', LabelField(dtype=torch.long)),
    ]
    return TextPreprocessor(fields, min_freq=3)


class BaselineModel(torch.nn.Module):
    def __init__(self, vocab_size=1, n_classes=1, pad_idx=0,
                 emb_dim=100, hidden_dim=256):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim, pad_idx)
        self._rnn = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self._out = torch.nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, inputs):
        question1, question2 = inputs
        q1, _ = self._rnn(self._emb(question1))[1]
        q2, _ = self._rnn(self._emb(question2))[1]

        hidden = torch.cat([q1.squeeze(0), q2.squeeze(0)], dim=-1)
        output = self._out(hidden)

        return output


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["question1"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__pad_idx=vocab["<pad>"])
        net.set_params(module__n_classes=len(X.fields["is_duplicate"].vocab))

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DeduplicationNet(skorch.NeuralNet):
    def predict(self, X):
        vocab = X.fields["is_duplicate"].vocab
        return np.take(vocab.itos, self.predict_proba(X).argmax(-1))


def build_model():
    model = DeduplicationNet(
        module=BaselineModel,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
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
    print(data())


if __name__ == '__main__':
    main()
