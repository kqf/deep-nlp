import torch
import random
import spacy
import numpy as np
import pandas as pd
import gensim.downloader as api

from tqdm import tqdm
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
        return [[t for t in self.spacy.tokenizer(text)] for text in texts]


class BatchIterator():
    def __init__(self, word2ind, data):
        self._data = data
        self._num_samples = len(data)
        self.word2ind = word2ind

    def buckets(self, batch_size, device, shuffle=True):
        batch_count = int(self._num_samples / batch_size)
        return self._iterate_batches(batch_size, device, shuffle), batch_count

    def _iterate_batches(self, batch_size, device, shuffle):
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
                'questions': self.to_matrix(questions).to(device),
                'correct_answers': self.to_matrix(correct_answers).to(device),
                'wrong_answers': self.to_matrix(wrong_answers).to(device),
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


class ModelTrainer():
    _msg = '{:>5s} Loss = {:.5f}, Recall@1 = {:.2%}'

    def __init__(self, model, optimizer):
        self._model = model
        self._optimizer = optimizer

    def on_epoch_begin(self, is_train, name, batches_count):
        """
        Initializes metrics
        """
        self._epoch_loss = 0
        self._correct_count = 0
        self._total_count = 0
        self._is_train = is_train
        self._name = name
        self._batches_count = batches_count

        self._model.train(is_train)

    def on_epoch_end(self):
        """
        Outputs final metrics
        """
        return self._msg.format(
            self._name,
            self._epoch_loss / self._batches_count,
            self._correct_count / self._total_count
        )

    def _loss(self, batch):
        return torch.tensor(0), 1, 0

    def on_batch(self, batch):
        loss, total_count, correct_count = self._loss(batch)

        if self._is_train:
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.)
            self._optimizer.step()

        return self._msg.format(
            self._name,
            loss.item(),
            correct_count / total_count
        )

    def epoch(self, data_iter, batches_count, is_train, name=None):
        self.on_epoch_begin(is_train, name, batches_count=batches_count)

        with torch.autograd.set_grad_enabled(is_train):
            with tqdm(total=batches_count) as progress_bar:
                for i, batch in enumerate(data_iter):
                    batch_progress = self.on_batch(batch)

                    progress_bar.update()
                    progress_bar.set_description(batch_progress)

                epoch_progress = self.on_epoch_end()
                progress_bar.set_description(epoch_progress)
                progress_bar.refresh()


class ChatModel(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=32, epochs_count=30):
        self.batch_size = batch_size
        self.epochs_count = epochs_count

    def _init_trainer(self, X, y=None):
        self.model = DSSM()
        optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer = ModelTrainer(self.model, optimizer)

    def fit(self, X, y=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._init_trainer(X, y)

        train, batches_count = X.buckets(
            batch_size=self.batch_size, device=device)

        for epoch in range(self.epochs_count):
            name = '[{} / {}] Train'.format(epoch + 1, self.epochs_count)
            self.trainer.epoch(train, batches_count, is_train=True, name=name)
        return self


class DSSM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = None
        self.correct = None
        self.wrong = None
        self.linear = torch.nn.Linear(10, 20)

    def forward(self, query_inputs, correct_inputs, wrong_inputs):
        batch_size = query_inputs.shape[0]
        return (
            torch.randint(0, 10, (batch_size,)),
            torch.randint(0, 10, (batch_size,)),
            torch.randint(0, 10, (batch_size,)),
        )


def similarity(a, b):
    return (a * b).sum(-1)


def triplet_loss(query, correct, wrong, delta=1.0):
    # loss = max(0, 1 - pos_sim + neg_sim)
    return torch.nn.functional.relu(
        delta - similarity(query, correct) + similarity(query, wrong)
    )


def build_vectorizer():
    w2v_model = api.load('glove-wiki-gigaword-100')
    vect = make_pipeline(
        Tokenizer("question", "options"),
        TextVectorizer(w2v_model),
    )
    return vect


def build_model():
    model = make_pipeline(
        build_vectorizer(),
        ChatModel(),
    )
    return model


def main():
    train, test = read_dataset()
    tt = Tokenizer("question", "options").fit_transform(train)
    print(tt.head())


if __name__ == '__main__':
    main()
