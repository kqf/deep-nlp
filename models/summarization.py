import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


"""
!curl -k -L "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab" -o data/news.zip
!pip install torch
!pip install torchtext
!pip install sacremoses
"""  # noqa

"""
    - [ ] Boilerplate
    - [ ] Decoder encoder
    - [ ] Transformer
"""


def data():
    return pd.read_csv("data/news.zip")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=3, max_tokens=16, bpe_col_prefix=None,
                 init_token="<s>", eos_token="</s>"):
        self.bpe_col_prefix = bpe_col_prefix
        self.min_freq = min_freq
        self.max_tokens = max_tokens
        self.source = "source"
        self.target = "target"
        self.text = Field(
            tokenize='moses',
            init_token=init_token, eos_token=eos_token, lower=True)
        self.fields = [(self.source, self.text), (self.target, self.text)]

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        self.text.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        sources = X[self.source].apply(self.text.preprocess)
        targets = X[self.target].apply(self.text.preprocess)

        valid_idx = (
            (sources.str.len() < self.max_tokens) & (
                targets.str.len() < self.max_tokens)
        )
        examples = [Example.fromlist(pair, self.fields)
                    for pair in zip(sources[valid_idx], targets[valid_idx])]
        dataset = Dataset(examples, self.fields)
        return dataset


def main():
    df = data()
    print(df.head())


if __name__ == '__main__':
    main()
