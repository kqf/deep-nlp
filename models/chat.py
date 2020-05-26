import torch
import random
import math
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

    for text in data["question"]:
        for word in text:
            words[word] += 1

    for options in data["options"]:
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


class BatchIterator():
    def __init__(self, data, batch_size, word2ind, shuffle=True):
        self._data = data
        self._num_samples = len(data)
        self._batch_size = batch_size
        self._word2ind = word2ind
        self._shuffle = shuffle
        self._batches_count = int(math.ceil(len(data) / batch_size))

    def __len__(self):
        return self._batches_count

    def __iter__(self):
        return self._iterate_batches()

    def _iterate_batches(self):
        indices = np.arange(self._num_samples)
        if self._shuffle:
            np.random.shuffle(indices)

        for start in range(0, self._num_samples, self._batch_size):
            end = min(start + self._batch_size, self._num_samples)

            batch_indices = indices[start: end]

            batch = self._data.iloc[batch_indices]
            questions = batch['question'].values
            correct_answers = np.array([
                row['options'][random.choice(row['correct_indices'])]
                for i, row in batch.iterrows()
            ])
            wrong_answers = np.array([
                row['options'][random.choice(row['wrong_indices'])]
                for i, row in batch.iterrows()
            ])

            yield {
                'questions': self.to_matrix(questions),
                'correct_answers': self.to_matrix(correct_answers),
                'wrong_answers': self.to_matrix(wrong_answers)
            }

    def to_matrix(self, lines):
        max_sent_len = max(len(line) for line in lines)
        matrix = np.zeros((len(lines), max_sent_len))

        for i, line in enumerate(lines):
            matrix[i, :len(line)] = [self.word2ind.get(w, 1) for w in line]
        return torch.LongTensor(matrix)


def main():
    train, test = read_dataset()
    print(train.head())


if __name__ == '__main__':
    main()
