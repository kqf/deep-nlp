import time
import torch
import skorch
import random
import numpy as np
import pandas as pd
from functools import partial
from contextlib import contextmanager
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.data import BucketIterator

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
                 hidden_size=256, classes_count=2, rnn=None):
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


def build_preprocessor():
    text_field = Field(batch_first=False, tokenize=lambda x: x)

    fields = [
        ("names", text_field),
        ("labels", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


def build_model(**kwargs):
    base_model = skorch.NeuralNetClassifier(
        module=RecurrentClassifier,
        module__vocab_size=1000,  # Dummy dimension
        module__emb_dim=24,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.01,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=BucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=BucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
        ],
    )
    model = make_pipeline(
        build_preprocessor(),
        base_model,
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
