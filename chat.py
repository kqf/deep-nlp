import spacy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_union


def read_dataset():
    train = pd.read_json('data/train.json')
    test = pd.read_json('data/test.json')
    return train, test


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.col].str.lower().apply(self.tokenize)

    @staticmethod
    def tokenize(text):
        return [t for t in spacy.tokenizer(text)]


def build_tokenizer(col1, col2):
    texts = make_union(
        Tokenizer(col1),
        Tokenizer(col2),
    )
    return texts


def main():
    train, test = read_dataset()

    tokenizer = build_tokenizer("question", "options")
    tokenizer.fit_transform(train)


if __name__ == '__main__':
    main()
