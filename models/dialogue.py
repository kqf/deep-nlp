import os
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
""" # noqa


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


def main():
    train, test, valid = data()
    print(train.head())


if __name__ == '__main__':
    main()
