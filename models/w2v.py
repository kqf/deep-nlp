import time
import math
import tqdm
import nltk
import torch
import random
import itertools
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import output_notebook

from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

"""

!mkdir -p data/
!pip install numpy torch pytest pandas kaggle nltk tqdm scikit-learn bokeh
!wget -O data/quora.zip -qq --no-check-certificate "https://drive.google.com/uc?export=download&id=1ERtxpdWOgGQ3HOigqAMHTJjmOE_tWvoF"
!unzip data/quora.zip -d data/
!zip  data/train.csv.zip  data/train.csv


import nltk
nltk.download('punkt')
"""  # noqa


def quora_data():
    df = pd.read_csv("data/train.csv.zip")
    df.replace(np.nan, '', regex=True, inplace=True)
    texts = list(pd.concat([df.question1, df.question2]).str.lower().unique())
    return texts


def flatten(sequence):
    return itertools.chain(*sequence)


class IdentityTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.tokenized_texts = X
        if isinstance(self.tokenized_texts[0], str):
            self.tokenized_texts = [s.split() for s in X]

        self.word2index = {
            w: i for i, w in enumerate(set(flatten(self.tokenized_texts)))
        }

        self.index2word = list(
            sorted(self.word2index.items(), key=lambda x: x[1]))

        return [
            [self.word2index.get(t, 0) for t in tokens]
            for tokens in self.tokenized_texts
        ]


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=5, verbose=True, unk_token=0):
        self.verbose = verbose
        self.min_count = min_count
        self.tokenized_texts = None
        self.index2word = None
        self.word2index = {
            '<unk>': unk_token
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            nltk.data.find('tokenizers/punkt.zip')
        except LookupError:
            nltk.download('punkt')

        self.tokenized_texts = [word_tokenize(t.lower()) for t in tqdm.tqdm(X)]
        flat = flatten(self.tokenized_texts)
        words_counter = Counter(flat)

        for word, count in words_counter.most_common():
            if count < self.min_count:
                break
            self.word2index[word] = len(self.word2index)

        self.index2word = [
            word for word, _ in
            sorted(self.word2index.items(), key=lambda x: x[1])]

        if self.verbose:
            print('Vocabulary size:', len(self.word2index))
            print('Tokens count:', sum(map(len, self.tokenized_texts)))
            print('Unknown tokens appeared:', len(
                set(flat) - set(self.word2index.keys())))
            print('Most freq words:', self.index2word[1:21])

        return [
            [self.word2index.get(t, 0) for t in tokens]
            for tokens in self.tokenized_texts
        ]


class SkipGram:
    def __init__(
        self,
        dim=32,
        window_size=2,
        num_skips=4,
        batch_size=128,
        lr=0.01,
        tokenizer=IdentityTokenizer(),
        loss_nsteps=1000,
        verbose=True
    ):
        self.dim = dim
        self.window_size = window_size
        self.num_skips = num_skips
        self.batch_size = batch_size
        self.lr = lr
        self.tokenizer = tokenizer
        self.loss_nsteps = loss_nsteps
        self.verbose = verbose
        self.model = None

    def fit(self, X):
        tokenized_texts = self.tokenizer.fit_transform(X)

        self.model = torch.nn.Sequential(
            torch.nn.Embedding(len(self.tokenizer.word2index), self.dim),
            torch.nn.Linear(self.dim, len(self.tokenizer.word2index))
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        total_loss = 0
        start_time = time.time()

        data = self.batches(
            self.build_contexts(tokenized_texts, window_size=self.window_size),
            window_size=self.window_size,
            num_skips=self.num_skips,
            batch_size=self.batch_size,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_function = torch.nn.CrossEntropyLoss().to(device)

        for step, (batch, labels) in enumerate(tqdm.tqdm(data)):
            batch = torch.LongTensor(batch).to(device)
            labels = torch.LongTensor(labels).to(device)

            logits = self.model(batch)

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if self.verbose and step != 0 and step % self.loss_nsteps == 0:
                print("Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(
                    step,
                    total_loss / self.loss_nsteps,
                    time.time() - start_time)
                )
                total_loss = 0
                start_time = time.time()
        return self

    @staticmethod
    def batches(contexts, window_size, num_skips, batch_size):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * window_size

        data = [
            (word, context) for word, context in contexts
            if len(context) == 2 * window_size and word != 0
        ]

        batch_size = int(batch_size / num_skips)
        batchs_count = int(math.ceil(len(data) / batch_size))

        print(f'Init batch-generator with {batchs_count} batchs per epoch')

        indices = np.arange(len(data))
        np.random.shuffle(indices)

        for batch_indices in np.array_split(indices, batchs_count):
            batch_data, batch_labels = [], []

            for idx in batch_indices:
                central_word, context = data[idx]

                words_to_use = random.sample(context, num_skips)
                batch_data.extend([central_word] * num_skips)
                batch_labels.extend(words_to_use)

            yield batch_data, batch_labels

    @staticmethod
    def build_contexts(tokenized_texts, window_size):
        for tokens in tokenized_texts:
            for i, central_word in enumerate(tokens):
                context = [
                    tokens[i + d] for d in range(-window_size, window_size + 1)
                    if d != 0 and 0 <= i + d < len(tokens)]
                yield central_word, context

    @property
    def embeddings_(self):
        return self.model[0].weight.cpu().data.numpy()


def most_similar(embeddings, index2word, word2index, word, n_words=10):
    word_emb = embeddings[word2index[word]]

    similarities = cosine_similarity([word_emb], embeddings)[0]
    top_n = np.argsort(similarities)[-n_words:]

    return [index2word[index] for index in reversed(top_n)]


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    output_notebook()

    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource(
        {'x': x, 'y': y, 'color': color, **kwargs})

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color',
                alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(
        tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2, verbose=100)
    return scale(tsne.fit_transform(word_vectors))


def visualize_embeddings(embeddings, index2word, word_count=1000):
    word_vectors = embeddings[1: word_count + 1]
    words = index2word[1: word_count + 1]

    word_tsne = get_tsne_projection(word_vectors)
    draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='green', token=words)


def main():
    df = quora_data()
    model = SkipGram(tokenizer=Tokenizer()).fit(df)
    print(model.embeddings_)
    print("Most similar words")
    print(most_similar(model.embeddings_, model.tokenizer.index2word,
                       model.tokenizer.word2index, 'warm'))

    print("Visualize the embeddings")
    visualize_embeddings(model.embeddings_, model.tokenizer.index2word)


if __name__ == '__main__':
    main()
