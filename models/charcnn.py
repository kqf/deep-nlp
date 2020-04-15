import torch
import numpy as np
from collections import Counter


class ConvClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, filters_count):
        super().__init__()

        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, filters_count, (3, 1))
        self._dropout = torch.nn.Dropout(0.2)
        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(kernel_size=(1, 15))
        self._out_layer = torch.nn.Linear(filters_count * 15, 1, bias=False)
        model = torch.nn.Sequential(
            self.embedding,
            self._conv,
            self._dropout,
            self._relu,
            self._max_pooling,
            self._out_layer,
        )
        self.total = model

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        self.batch_size = inputs[0]
        return self.total(inputs)

    def embedding(self, inputs):
        batch_size, max_len = inputs.shape[0]
        embed = self._embedding(inputs).view(batch_size, 1, max_len, -1)
        return embed

    def max_pooling(self, inputs):
        # Original inputs shape
        return inputs.reshape(self.batch_size, -1)

    def get_filters(self):
        return self._conv.weight.data.cpu().detach().numpy()


class Tokenizer:
    def fit(self, X):
        chars = set("".join(X))
        self.char_index = {c: i + 1 for i, c in enumerate(chars)}
        self.char_index['<pad>'] = 0

        word_len_counter = Counter(list(map(len, X)))

        threshold = 0.99
        self.max_len = self._find_max_len(word_len_counter, threshold)
        return self

    @staticmethod
    def _find_max_len(counter, threshold):
        sum_count = sum(counter.values())
        cum_count = 0
        for i in range(max(counter)):
            cum_count += counter[i]
            if cum_count > sum_count * threshold:
                return i
        return max(counter)


def main():
    model = ConvClassifier()
    print(model)


if __name__ == '__main__':
    main()
