import time
import math
import torch
import random
import numpy as np
import pandas as pd
from functools import partial
from contextlib import contextmanager
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from torchtext.data import Dataset, Example, Field, LabelField

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{color}[{name}] done in {et:.0f} s{nocolor}".format(
        name=name, et=time.time() - t0,
        color='\033[1;33m', nocolor='\033[0m'))


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


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        return Dataset(examples, self.fields)


class SimpleRNNModel(torch.nn.Module):
    bidirectional = False

    def __init__(self, input_size, hidden_size, activation=None):
        super().__init__()

        self._hidden_size = hidden_size
        # Convention: X[batch, inputs] * W[inputs, outputs]
        self._hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self._activate = activation or torch.nn.ReLU()

    def forward(self, inputs, hidden=None):
        # RNN Convention: X[sequence, batch, inputs]
        seq_len, batch_size = inputs.shape[:2]

        if hidden is None:
            hidden = inputs.new_zeros((seq_len, batch_size, self._hidden_size))

        for i in range(seq_len):
            layer_input = torch.cat((hidden[(i - 1) * (i > 0)], inputs[i]), 1)
            hidden[i] = self._activate(self._hidden(layer_input))
        return hidden, None


def bilstm_out(x, backward=False):
    idx = int(backward)
    hidden_size = x.shape[-1]
    if hidden_size % 2 != 0:
        raise RuntimeError(f"Output of Bi-LSTM should be multiple of 2")

    shaped = x.reshape(x.shape[:-1] + (int(hidden_size / 2), 2))
    return shaped[:, :, idx]


class RecurrentClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 hidden_size, classes_count, rnn=None):
        super().__init__()
        self.classes_count = classes_count
        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        model_type = rnn or torch.nn.LSTM
        self._rnn = model_type(emb_dim, hidden_size)
        self._output = torch.nn.Linear(
            hidden_size + hidden_size * int(self._rnn.bidirectional),
            self.classes_count
        )

    def forward(self, inputs):
        embeded = self.embed(inputs)
        lstm_outputs, _ = self._rnn(embeded)

        recurrent = lstm_outputs[-1]
        if self._rnn.bidirectional:
            first = bilstm_out(lstm_outputs[0], backward=False)
            last = bilstm_out(lstm_outputs[-1], backward=True)
            recurrent = torch.cat([first, last], dim=-1)

        return self._output(recurrent)

    def embed(self, inputs):
        return self._embedding(inputs)


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


class CharClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 emb_dim=100,
                 hidden_size=20,
                 activation=None,
                 batch_size=128,
                 epochs_count=50,
                 rnn=None,
                 print_frequency=10):

        self.rnn = rnn
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.activation = activation
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.print_frequency = print_frequency

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.model = RecurrentClassifier(
            vocab_size=len(np.unique(X)),
            emb_dim=self.emb_dim,
            hidden_size=self.hidden_size,
            classes_count=len(np.unique(y)),
            rnn=self.rnn,
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        batchs_count = int(math.ceil(len(X) / self.batch_size))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        total_loss = 0
        for epoch in range(self.epochs_count):
            for batch_indices in np.array_split(indices, batchs_count):
                X_batch, y_batch = X[batch_indices], y[batch_indices]
                # Convention all RNNs: [integer sequence, batch]
                x_rnn = X_batch.T

                batch = torch.LongTensor(x_rnn).to(device)
                labels = torch.LongTensor(y_batch).to(device)

                optimizer.zero_grad()

                self.model.eval()
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

                total_loss += loss.item()
            self._status(loss, epoch)

        return self

    def _status(self, loss, epoch=-1):
        if (epoch + 1) % self.print_frequency != 0:
            return
        self.model.eval()

        with torch.no_grad():
            msg = '[{}/{}] Training loss: {:.3f}'
            print(msg.format(
                epoch + 1,
                self.epochs_count,
                loss / self.epochs_count)
            )

    def predict_proba(self, X):
        X = np.array(X)
        self.model.eval()

        # Convention all RNNs: [sequence, batch, input_size]
        x_rnn = X.T
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch = torch.LongTensor(x_rnn).to(device)
        with torch.no_grad():
            preds = torch.nn.functional.softmax(self.model(batch), dim=-1)
        return preds.detach().cpu().data.numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)


def build_model(**kwargs):
    model = make_pipeline(
        Tokenizer(),
        CharClassifier(**kwargs),
    )
    return model


def baseline_model():
    model = make_pipeline(
        CountVectorizer(analyzer='char', ngram_range=(1, 4)),
        LogisticRegression(),
    )
    return model


def main():

    torch.manual_seed(42)
    np.random.seed(42)

    df = data()
    le = LabelEncoder()
    X, y = df["surname"], le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = baseline_model()
    with timer("Fit logistic regression"):
        model.fit(X_tr, y_tr)
    print("Logistic Regression:")
    print("Train score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print("Test score", f1_score(model.predict(X_tr), y_tr, average="micro"))

    model = build_model()
    with timer("Fit fit the LSTM"):
        model.fit(X_tr, y_tr)

    print("LSTM:")
    print("Train score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print("Test score", f1_score(model.predict(X_tr), y_tr, average="micro"))

    print("SimpleRNN:")
    model = build_model(rnn=SimpleRNNModel)
    with timer("Fit fit the simple RNN"):
        model.fit(X_tr, y_tr)

    print("Train score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print("Test score", f1_score(model.predict(X_tr), y_tr, average="micro"))

    print("Bidirectional LSTM:")
    model = build_model(rnn=partial(torch.nn.LSTM, bidirectional=True))
    with timer("Fit fit the simple RNN"):
        model.fit(X_tr, y_tr)

    print("Train score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print("Test score", f1_score(model.predict(X_tr), y_tr, average="micro"))


if __name__ == '__main__':
    main()
