import math
import torch
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
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


class RecurrentClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, classes_count):
        super().__init__()
        self.classes_count = classes_count
        self._embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self._output = torch.nn.Linear(vocab_size, self.classes_count)

        # <set layers >

    def forward(self, inputs):
        # 'embed(inputs) -> prediction'
        # <implement it >
        embeded = self.embed(inputs)
        return self._output(inputs.squeeze(-1).T.float())

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
                 hidden_size=100,
                 activation=None,
                 batch_size=100,
                 epochs_count=50,
                 print_frequency=10):

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
            classes_count=len(np.unique(y))
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
                # Convention all RNNs: [sequence, batch, input_size]
                x_rnn = X_batch.T[:, :, np.newaxis]

                batch = torch.LongTensor(x_rnn).to(device)
                labels = torch.LongTensor(y_batch).to(device)

                optimizer.zero_grad()

                self.model.eval()
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()

                total_loss += loss.item()
            self._status(loss, epoch)

        return self

    def _status(self, loss, epoch=-1):
        if (epoch + 1) % self.print_frequency != 0:
            return
        self.model.eval()

        with torch.no_grad():
            msg = '[{}/{}] Train: {:.3f}'
            print(msg.format(
                epoch + 1,
                self.epochs_count,
                loss / self.epochs_count)
            )

    def predict_proba(self, X):
        X = np.array(X)
        self.model.eval()

        # Convention all RNNs: [sequence, batch, input_size]
        x_rnn = X.T[:, :, np.newaxis]
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
