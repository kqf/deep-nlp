import torch
import random
import spacy
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


def read_dataset():
    train = pd.read_json('data/train.json').dropna()
    test = pd.read_json('data/test.json').dropna()
    return train, test


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, question, options, language='en'):
        self.spacy = spacy.load(language)
        self.question = question
        self.options = options

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = pd.DataFrame(X)
        output.loc[:, self.question] = X[self.question].apply(self.tokenize)
        output.loc[:, self.options] = X[self.options].apply(self.otokenize)
        return output

    def tokenize(self, text):
        return [t for t in self.spacy.tokenizer(text)]

    def otokenize(self, texts):
        return [t for text in texts for t in self.spacy.tokenizer(text)]


class BatchIterator():
    def __init__(self, word2ind, data):
        self._data = data
        self._num_samples = len(data)
        self._word2ind = word2ind

    def buckets(self, batch_size, shuffle=True):
        return self._iterate_batches(batch_size, shuffle)

    def _iterate_batches(self, batch_size, shuffle):
        indices = np.arange(self._num_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self._num_samples, batch_size):
            end = min(start + batch_size, self._num_samples)

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

class TextVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model, min_freq=5,
                 pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        self.w2v_model = w2v_model
        self.word2ind = None

    def fit(self, X):
        words = Counter()

        for text in X["question"]:
            for word in text:
                words[word] += 1

        for options in X["options"]:
            for text in options:
                for word in text:
                    words[word] += 1

        self.word2ind = {
            self.pad_token: 0,
            self.unk_token: 1
        }

        embeddings = [
            np.zeros(self.w2v_model.vectors.shape[1]),
            np.zeros(self.w2v_model.vectors.shape[1])
        ]

        for word, count in words.most_common():
            if count < self.min_freq:
                break

            if word not in self.w2v_model.vocab:
                continue

            self.word2ind[word] = len(self.word2ind)
            embeddings.append(self.w2v_model.get_vector(word))

        self.embeddings = np.array(embeddings)
        return self

    def transform(self, X):
        return BatchIterator(self.word2ind, X)


def main():
    train, test = read_dataset()
    tt = Tokenizer("question", "options").fit_transform(train)
    print(tt.head())


if __name__ == '__main__':
    main()
