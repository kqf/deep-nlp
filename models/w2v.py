import math
import random
import tqdm
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter


def quora_data():
    df = pd.read_csv("data/train.csv.zip")
    df.replace(np.nan, '', regex=True, inplace=True)
    texts = list(pd.concat([df.question1, df.question2]).str.lower().unique())
    return texts


class Tokenizer:
    def __init__(self, min_count=5, verbose=True, unk_token=0):
        self.verbose = verbose
        self.min_count = min_count
        self.tokenized_texts = None
        self.index2word = None
        self.word2index = {
            '<unk>': unk_token
        }

    def fit(self, X):
        try:
            nltk.data.find('tokenizers/punkt.zip')
        except LookupError:
            nltk.download('punkt')

        self.tokenized_texts = [word_tokenize(t.lower()) for t in tqdm.tqdm(X)]
        words_counter = Counter(sum(self.tokenized_texts, []))

        for word, count in words_counter.most_common():
            print(count)
            if count < self.min_count:
                break
            self.word2index[word] = len(self.word2index)

        self.index2word = list(
            sorted(self.word2index.items(), key=lambda x: x[1]))

        if self.verbose:
            print('Vocabulary size:', len(self.word2index))
            print('Tokens count:', sum(map(len, self.tokenized_texts)))
            print('Unknown tokens appeared:', len(
                set(sum(self.tokenized_texts, [])) -
                set(self.word2index.keys())))
            print('Most freq words:', self.index2word[1:21])
        return self


def build_contexts(tokenized_texts, window_size):
    for tokens in tokenized_texts:
        for i, central_word in enumerate(tokens):
            context = [
                tokens[i + di] for di in range(-window_size, window_size + 1)
                if di != 0 and 0 <= i + di < len(tokens)]
            yield central_word, context


def skip_gram_batchs(contexts, window_size, num_skips, batch_size):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size

    data = [
        (word, context) for word, context in contexts
        if len(context) == 2 * window_size and word != 0
    ]

    batch_size = int(batch_size / num_skips)
    batchs_count = int(math.ceil(len(data) / batch_size))

    print(f'Initializing batch-generator with {batchs_count} batchs per epoch')

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


def main():
    df = quora_data()
    tokenizer = Tokenizer().fit(df)


if __name__ == '__main__':
    main()
