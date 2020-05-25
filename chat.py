import spacy
import numpy as np
import pandas as pd

from collections import Counter
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


def build_word_embeddings(data, w2v_model, min_freq=5):
    words = Counter()

    for text in data.question:
        for word in text:
            words[word] += 1

    for options in data.options:
        for text in options:
            for word in text:
                words[word] += 1

    word2ind = {
        '<pad>': 0,
        '<unk>': 1
    }

    embeddings = [
        np.zeros(w2v_model.vectors.shape[1]),
        np.zeros(w2v_model.vectors.shape[1])
    ]

    for word, count in words.most_common():
        if count < min_freq:
            break

        if word not in w2v_model.vocab:
            continue

        word2ind[word] = len(word2ind)
        embeddings.append(w2v_model.get_vector(word))

    return word2ind, np.array(embeddings)


def main():
    train, test = read_dataset()

    tokenizer = build_tokenizer("question", "options")
    tokenizer.fit_transform(train)


if __name__ == '__main__':
    main()
