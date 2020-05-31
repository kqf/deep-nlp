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
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


def build_preprocessor():
    text_field = Field(lower=True)
    fields = [
        ('query', text_field),
        ('target', text_field),
    ]
    return TextPreprocessor(fields, need_vocab={text_field: 5})


def main():
    train, test, val = data()
    print(train.head())


if __name__ == '__main__':
    main()
