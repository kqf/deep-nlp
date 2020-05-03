import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import nltk
np.random.seed(42)


class Tokenizer():

    def __init__(self):
        self.word2ind = None
        self.tag2ind = None

    def fit(self, X, y=None):
        words = {word for sample in X for word, tag in sample}
        self.word2ind = {word: ind + 1 for ind, word in enumerate(words)}
        self.word2ind['<pad>'] = 0

        tags = {tag for sample in X for word, tag in sample}
        self.tag2ind = {tag: ind + 1 for ind, tag in enumerate(tags)}
        self.tag2ind['<pad>'] = 0
        return self

    def transform(self, X, none):
        X = [[self.word2ind.get(word, 0) for word, _ in sample]
             for sample in X]
        y = [[self.tag2ind[tag] for _, tag in sample] for sample in X]
        return X, y


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')

    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    for i, d in enumerate(data):
        print(i, d)
        if i > 5:
            break


if __name__ == '__main__':
    main()
