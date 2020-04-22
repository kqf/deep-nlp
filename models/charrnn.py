import math
# import time
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super().__init__()

        self._hidden_size = hidden_size
        # Convention: X[batch, inputs] * W[inputs, outputs]
        self._hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self._activate = activation or torch.nn.Tanh()

    def forward(self, inputs, hidden=None):
        # RNN Convention: X[sequence, batch, inputs]
        seq_len, batch_size = inputs.shape[:2]

        if hidden is None:
            hidden = inputs.new_zeros((batch_size, self._hidden_size))

        for i in range(seq_len):
            layer_input = torch.cat((hidden, inputs[i]), dim=1)
            hidden = self._activate(self._hidden(layer_input))
        return hidden


class MemorizerModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, activation=None):
        super().__init__()

        self._hidden_size = hidden_size
        self._embedding = torch.nn.Embedding.from_pretrained(
            torch.eye(embedding_size, requires_grad=True))
        self._rnn = SimpleRNN(embedding_size, hidden_size, activation)
        self._linear = torch.nn.Linear(hidden_size, 1)

        self._model = torch.nn.Sequential(
            self._embedding,
            self._rnn,
            self._linear

        )

    def forward(self, inputs, hidden=None):
        # Convention: inputs[sequence, batch, input_size=1]
        return self._model(inputs.squeeze(dim=-1))


def generate_data(num_batches=10, batch_size=25, seq_len=5):
    for _ in range(num_batches * batch_size):
        data = np.random.randint(0, 10, seq_len)
        yield data, data[0]


class BasicRNNClassifier():
    def __init__(self, hidden_size=100, batch_size=25, epochs_count=1):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs_count = epochs_count

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        rnn = MemorizerModel(10, hidden_size=self.hidden_size)
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


def baseline(X_tr, X_te, y_tr, y_te):
    model = make_pipeline(
        CountVectorizer(analyzer='char', ngram_range=(1, 4)),
        LogisticRegression(),
    )
    model.fit(X_tr, y_tr)

    print("Train")
    print(classification_report(y_tr, model.predict(X_tr)))

    print("Test")
    print(classification_report(y_te, model.predict(X_te)))
    return model


def main():
    df = data()
    print(df)
    le = LabelEncoder()
    X, y = df["surname"], le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    base = baseline(X_tr, X_te, y_tr, y_te)
    print(base)


if __name__ == '__main__':
    main()
