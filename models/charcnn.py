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
        self.c2i = {c: i + 1 for i, c in enumerate(chars)}
        self.c2i['<pad>'] = 0

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

    def transform(self, X):
        shorted_data = []
        for word in X:
            cc = np.array([self.c2i.get(s, 0) for s in word[:self.max_len]])
            padded = np.pad(cc, (0, self.max_len - len(cc)), mode="constant")
            shorted_data.append(padded)
        return np.array(shorted_data)


class CharClassifier:

    @staticmethod
    def batches(X, y, batch_size):
        num_samples = X.shape[0]

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_idx = indices[start: end]
            yield X[batch_idx], y[batch_idx]


def main():
    model = ConvClassifier()
    print(model)


if __name__ == '__main__':
    main()
