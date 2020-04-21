import time
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


class LSTMModel(torch.nn.Module):
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


class CharLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, batch_size=32, epochs_count=1, verbose=True):
        self.batch_size = batch_size
        self.epochs_count = epochs_count
        self.verbose = verbose
        self.model = None
        self.lr = lr
        self.clsses_ = [0, 1]

    @staticmethod
    def batches(X, y, batch_size):
        num_samples = X.shape[0]

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_idx = indices[start: end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X, y):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = LSTMModel(1000)

        optimizer = torch.optim.Adam(
            [param for param in self.model.parameters()
             if param.requires_grad],
            lr=self.lr)

        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs_count):
            batches = self.batches(X, y, self.batch_size)
            epoch_loss = 0
            time_epoch = time.time()
            for i, (X_batch, y_batch) in enumerate(batches):
                Xt = torch.LongTensor(X_batch).to(device)
                yt = torch.LongTensor(y_batch).to(device)

                logits = self.model.forward(Xt)
                loss = loss_function(logits, yt)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                te = time.time() - time_epoch
                print(f"Epoch {epoch}, loss {epoch_loss}, time {te:.2f}")
        return self

    def predict_proba(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.LongTensor(X).to(device)
        logits = self.model(X)
        return torch.sigmoid(logits).cpu().data.numpy()

    def embeddings(self, X):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = torch.LongTensor(X).to(device)
        return self.model.embed(X).cpu().data.numpy()

    def predict(self, X):
        self.model.eval()
        return self.predict_proba(X).argmax(axis=1)


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
