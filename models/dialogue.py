import os
import pandas as pd
from torchtext.data import LabelField, Field, Example, Dataset
from sklearn.base import BaseEstimator, TransformerMixin

# from torchtext.data import Dataset, BucketIterator


"""
!mkdir -p data
!git clone https://github.com/MiuLab/SlotGated-SLU.git
!mv SlotGated-SLU data
"""


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
    def __init__(self, fields, min_freq=3):
        self.min_freq = min_freq
        self.fields = fields

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for (_, field) in self.fields:
            field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        cols = [X[col].apply(field.preprocess) for col, field in self.fields]
        examples = [Example.fromlist(c, self.fields) for c in zip(cols)]
        return Dataset(examples, self.fields)


def build_preprocessor():
    fields = [
        ('tokens', Field()),
        ('tags', Field(unk_token=None)),
        ('intent', LabelField()),
    ]
    return TextPreprocessor(fields)


def main():
    train, test, valid = data()
    print(train.head())


if __name__ == '__main__':
    main()