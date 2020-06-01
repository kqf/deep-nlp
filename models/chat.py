import torch
import skorch
import pandas as pd
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.base import BaseEstimator, TransformerMixin


def read_data(filename):
    with open(filename) as f:
        data = f.readlines()
    return pd.DataFrame(
        list(zip(data[:-1], data[1:])),
        columns=["query", "target"])


def data():
    return (
        read_data("data/train.txt"),
        read_data("data/test.txt"),
        read_data("data/valid.txt"),
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, need_vocab=None):
        self.fields = fields
        self.need_vocab = need_vocab or {}

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for field, min_freq in self.need_vocab.items():
            field.build_vocab(dataset, min_freq=min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col.replace("_dummy", "")].apply(f.preprocess)
                for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._out = torch.nn.Linear(100, 20)

    def forward(self, inputs):
        return inputs


class TripletLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(100).mean()

    def forward(self, first, second):
        return ((first - second) ** 2).mean()


def build_preprocessor():
    text_field = Field(lower=True)
    target_field = Field(lower=True, is_target=True)
    fields = [
        ('query', text_field),
        ('target', text_field),
        # ('target_dummy', target_field)
    ]
    return TextPreprocessor(fields, need_vocab={
        text_field: 5, target_field: 5})


class UnsupervisedNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        query, target = y_pred
        return self.criterion_(query, target)


def build_model():
    model = UnsupervisedNet(
        module=SimpleModule,
        criterion=TripletLoss,
        iterator_train=BucketIterator,
        iterator_valid=BucketIterator,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
    )
    return model


def main():
    train, test, val = data()
    print(train.head())


if __name__ == '__main__':
    main()
