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

    def __init__(self, input_size, hid_size, activation=None):
        super().__init__()

        self._hidden_size = hid_size
        # Convention: X[batch, inputs] * W[inputs, outputs]
        self._hidden = torch.nn.Linear(input_size + hid_size, hid_size)
        self._activate = activation or torch.nn.ReLU()

    def forward(self, inputs, hidden=None):
        # RNN Convention: X[sequence, batch, inputs]
        seq_len, batch_size = inputs.shape[:2]

        if hidden is None:
            hidden = inputs.new_zeros((seq_len, batch_size, self._hidden_size))

        for i in range(seq_len):
            layer_input = torch.cat((hidden[(i - 1) * (i > 0)], inputs[i]), 1)
            hidden[i] = self._activate(self._hidden(layer_input))
        return hidden, (hidden[-1].unsqueeze(0), None)


def context(inputs, bidirectional=False):
    if bidirectional:
        return torch.cat([inputs[-1, :, :], inputs[-2, :, :]], dim=-1)
    return inputs[-1, :, :]


class RecurrentClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 hid_size=256, classes_count=2, rnn_type=None):
        super().__init__()
        self.classes_count = classes_count
        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = rnn_type(emb_dim, hid_size)
        n_directins = int(self._rnn.bidirectional) + 1
        self._output = torch.nn.Linear(
            hid_size * n_directins,
            self.classes_count
        )

    def forward(self, inputs):
        embeded = self.embed(inputs)
        _, (hidden, _) = self._rnn(embeded)
        return self._output(context(hidden, self._rnn.bidirectional))

    def embed(self, inputs):
        return self._embedding(inputs)


def build_preprocessor():
    text_field = Field(batch_first=False, tokenize=lambda x: x)

    fields = [
        ("names", text_field),
        ("labels", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


class ClassificationParamSetter(skorch.callbacks.Callback):
    def __init__(self, textfield, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textfield = textfield

    def on_train_begin(self, net, X, y):
        vocab = X.fields[self.textfield].vocab
        net.set_params(module__vocab_size=len(vocab))

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(rnn_type=SimpleRNNModel):
    base_model = skorch.NeuralNetClassifier(
        module=RecurrentClassifier,
        module__rnn_type=rnn_type,
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
            ClassificationParamSetter("names"),
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


def evaluate(model_name, model, X_tr, y_tr, X_te, y_te):
    with timer(f"Fit {model_name}"):
        model.fit(X_tr, y_tr)

    print(f"\nEvaluate: {model_name}")
    print("Train score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print("Test score", f1_score(model.predict(X_tr), y_tr, average="micro"))
    print()

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    df = data()
    le = LabelEncoder()
    X, y = df["surname"], le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    evaluate("logistic regression", baseline_model(), X_tr, y_tr, X_te, y_te)
    evaluate("simple rnn", build_model(), X_tr, y_tr, X_te, y_te)
    evaluate("lstm", build_model(torch.nn.LSTM), X_tr, y_tr, X_te, y_te)

    bilstm = partial(torch.nn.LSTM, bidirectional=True)
    evaluate("bi-lstm", build_model(bilstm), X_tr, y_tr, X_te, y_te)


if __name__ == '__main__':
    main()
