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
        self.word2index = {
            '<unk>': unk_token
        }

    def fit(self, X):
        try:
            nltk.data.find('tokenizers/punkt.zip')
        except LookupError:
            nltk.download('punkt')

        tokenized_texts = [word_tokenize(t.lower()) for t in tqdm.tqdm(X)]
        words_counter = Counter(sum(tokenized_texts, []))

        for word, count in words_counter.most_common():
            print(count)
            if count < self.min_count:
                break
            self.word2index[word] = len(self.word2index)

        self.index2word = list(
            sorted(self.word2index.items(), key=lambda x: x[1]))

        if self.verbose:
            print('Vocabulary size:', len(self.word2index))
            print('Tokens count:', sum(map(len, tokenized_texts)))
            print('Unknown tokens appeared:', len(
                set(sum(tokenized_texts, [])) - set(self.word2index.keys())))
            print('Most freq words:', self.index2word[1:21])
        return self


def build_contexts(tokenized_texts, window_size):
    for tokens in tokenized_texts:
        for i, central_word in enumerate(tokens):
            context = [
                tokens[i + di] for di in range(-window_size, window_size + 1)
                if di != 0 and 0 <= i + di < len(tokens)]
            yield central_word, context


def main():
    df = quora_data()
    tokenizer = Tokenizer().fit(df)


if __name__ == '__main__':
    main()
