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

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        outputs = self.embed(inputs)
        return self._out_layer(outputs).squeeze(-1)

    def embed(self, inputs):
        model = torch.nn.Sequential(
            self._embedding,
            self._conv,
            self._dropout,
            self._relu,
            self._max_pooling,
        )
        return model(inputs)


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

    def fit(self, X, y, epochs_count=1,
            batch_size=32, val_data=None, val_batch_size=None):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = Tokenizer().fit(X)
        X = self.tokenizer.transform(X)

        self.model = ConvClassifier(len(self.tokenizer.c2i), 24, 64).to(device)
        optimizer = torch.optim.Adam(
            [param for param in self.model.parameters()
             if param.requires_grad],
            lr=0.01)

        for epoch in range(epochs_count):
            batches = self.batches(X, y, batch_size)
            epoch_loss = 0
            for i, (X_batch, y_batch) in enumerate(batches):
                X_batch = torch.LongTensor(X_batch)
                y_batch = torch.FloatTensor(y_batch)

                logits = self.model(X_batch)
                loss = torch.nn.CrossEntropyLoss(logits, y_batch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if is_train:
                #     < how to optimize the beast?>

                # <u can move the stuff to some function >
                # tp= < calc true positives >
                # fp= < calc false positives >
                # fn= < calc false negatives >

                # precision=...
                # recall=...
                # f1=...

                # epoch_tp += tp
                # epoch_fp += fp
                # epoch_fn += fn
        return self


def main():
    model = ConvClassifier()
    print(model)


if __name__ == '__main__':
    main()
