import torch
import numpy as np
from collections import Counter


class ConvClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, word_size=22, filters_count=3):
        super().__init__()

        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._conv = torch.nn.Conv2d(1, 1, (filters_count, 1))
        self._relu = torch.nn.ReLU()
        self._max_pooling = torch.nn.MaxPool2d(
            kernel_size=(word_size - filters_count + 1, 1))
        self._out_layer = torch.nn.Linear(emb_dim, 2, bias=False)

    def forward(self, inputs):
        '''
        inputs - LongTensor with shape (batch_size, max_word_len)
        outputs - FloatTensor with shape (batch_size,)
        '''
        outputs = self.embed(inputs)
        return self._out_layer(outputs).squeeze(1).squeeze(1)

    def embed(self, inputs):

        embs = self._embedding(inputs)
        model = torch.nn.Sequential(
            self._conv,
            self._relu,
            self._max_pooling,
        )
        return model(embs.unsqueeze(dim=1))


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


def custom_f1(y_pred, y):
    positives = y_pred.astype(bool)
    tp = np.sum(y_pred[positives] & y[positives])
    fp = np.sum(y_pred[positives] & 1 - y[positives])
    fn = np.sum(y_pred[~positives] | y[~positives])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)
    return f1 if np.isfinite(f1) else 0


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
            batch_size=32, val_data=None, val_batch_size=None, verbose=True):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = Tokenizer().fit(X)
        X = self.tokenizer.transform(X)
        print(X)

        self.model = ConvClassifier(
            len(self.tokenizer.c2i),
            emb_dim=24,
            word_size=X.shape[1],
            filters_count=3).to(device)

        optimizer = torch.optim.Adam(
            [param for param in self.model.parameters()
             if param.requires_grad],
            lr=0.01)

        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs_count):
            batches = self.batches(X, y, batch_size)
            epoch_loss = 0
            epoch_f1 = 0
            for i, (X_batch, y_batch) in enumerate(batches):
                Xt = torch.LongTensor(X_batch).to(device)
                yt = torch.LongTensor(y_batch).to(device)

                logits = self.model.forward(Xt)
                loss = loss_function(logits, yt)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_f1 += custom_f1(self.predict(X_batch), y_batch)

            if verbose:
                print(f"Epoch {epoch}, F1 {epoch_f1}")
        return self

    def predict_proba(self, X):
        X = self.tokenizer.transform(X)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.LongTensor(X).to(device)
        logits = self.model(X)
        return torch.nn.functional.softmax(logits, dim=1)

    def predict(self, X):
        return self.predict_proba(X).cpu().data.numpy().argmax(axis=1)


def main():
    model = ConvClassifier()
    print(model)


if __name__ == '__main__':
    main()
