import math
# import time
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.metrics import classification_report


"""
!mkdir -p data
!pip3 -qq install torch
!pip install -qq bokeh
!pip install -qq eli5
!pip install scikit-learn
!curl -k -L "https://drive.google.com/uc?export=download&id=1ji7dhr9FojPeV51dDlKRERIqr3vdZfhu" -o data/surnames-multilang.txt
"""  # noqa


def data(filename="data/surnames-multilang.txt"):
    return pd.read_csv(filename, sep="\t", names=["surname", "label"])


class MemorizerModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._embedding = torch.nn.Embedding.from_pretrained(
            torch.eye(10, requires_grad=True).float())
        self._hidden_layer = torch.nn.Linear(10 + hidden_size, hidden_size)
        self._relu = torch.nn.LeakyReLU()
        self._linear = torch.nn.Linear(hidden_size, 10)

    def forward(self, inputs, hidden=None):
        seq_len, batch_size = inputs.shape[:2]

        embed = self._embedding(inputs)
        if hidden is None:
            hidden = embed.new_zeros((batch_size, self._hidden_size)).float()

        for i in range(seq_len):
            layer_input = torch.cat((embed[i], hidden), 1)
            hidden = self._relu(self._hidden_layer(layer_input))
        result = self._linear(hidden)
        return result


class BasicRNNClassifier():
    def __init__(self, hidden_size=100, batch_size=25, epochs_count=1):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs_count = epochs_count

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        rnn = MemorizerModel(hidden_size=self.hidden_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, rnn.parameters()))

        total_loss = 0
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        batchs_count = int(math.ceil(len(X) / self.batch_size))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for epoch_ind in range(self.epochs_count):
            for batch_indices in np.array_split(indices, batchs_count):
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                batch = torch.LongTensor(X_batch).to(device)
                labels = torch.LongTensor(y_batch).to(device)

                optimizer.zero_grad()
                rnn.train()

                logits = rnn(batch)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        return self


class Tokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        chars = set("".join(X))
        self.c2i = {c: i + 1 for i, c in enumerate(chars)}
        self.c2i['<pad>'] = 0
        self.max_len = max(map(len, X))
        return self

    def transform(self, X):
        padded_data = []
        for word in X:
            cc = np.array([self.c2i.get(s, 0) for s in word[:self.max_len]])
            padded = np.pad(cc, (0, self.max_len - len(cc)), mode="constant")
            padded_data.append(padded)
        return np.array(padded_data)


def build_model(**kwargs):
    model = make_pipeline(
        Tokenizer(),
        CharClassifier(**kwargs),
    )
    return model


def main():
    df = data()
    le = LabelEncoder()
    X, y = df["surname"], le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_tr, y_tr)
    print("Train score", model.score(X_tr, y_tr))
    print("Test score", model.score(X_te, y_te))


if __name__ == '__main__':
    main()
