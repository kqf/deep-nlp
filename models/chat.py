import torch
import skorch
import random
import numpy as np
import pandas as pd
from operator import attrgetter
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


"""
!mkdir -p data
!pip -qq install torch
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
"""  # noqa

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def read_data(filename):
    with open(filename) as f:
        data = f.readlines()
    return pd.DataFrame(
        list(zip(data[:-1], data[1:])),
        columns=["query", "target"])


def data():
    return (
        read_data("data/train.txt"),
        read_data("data/test.txt"),
        read_data("data/valid.txt"),
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, need_vocab=None):
        self.fields = fields
        self.need_vocab = need_vocab or {}

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for field, min_freq in self.need_vocab.items():
            field.build_vocab(dataset, min_freq=min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


class DSSMEncoder(torch.nn.Module):
    def __init__(self, embeddings, hidden_dim=128, output_dim=128):
        super().__init__()
        # self._embs = torch.nn.Embedding.from_pretrained(
        #     torch.FloatTensor(embeddings))
        # emb_dim = embeddings.shape[1]
        emb_dim = 100
        self._embs = torch.nn.Embedding(embeddings, emb_dim)

        self._conv = torch.nn.Sequential(
            torch.nn.Conv1d(emb_dim, hidden_dim, kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )
        self._out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        embs = self._embs(inputs.T)
        embs = embs.permute(0, 2, 1)

        outputs = self._conv(embs)
        outputs = torch.max(outputs.T, -1)[0]
        return self._out(outputs)


class DSSM(torch.nn.Module):
    def __init__(self, embeddings=100):
        super().__init__()
        self.query = DSSMEncoder(embeddings)
        self.target = DSSMEncoder(embeddings)

    def forward(self, query, target):
        return self.query(query), self.target(target)


def cosine(a, b):
    unit_a = a / a.norm(p=2, dim=-1, keepdim=True)
    unit_b = b / b.norm(p=2, dim=-1, keepdim=True)
    return (unit_a * unit_b).sum(-1)


class TripletLoss(torch.nn.Module):
    def __init__(self, sim=cosine, delta=1.0):
        super().__init__()
        self.delta = delta
        self.sim = sim

    def forward(self, queries, targets):
        wrong = self.negatives(queries, targets)
        correct = targets
        return torch.nn.functional.relu(
            self.delta - self.sim(queries, correct) + self.sim(queries, wrong)
        ).mean()

    def negatives(self, a, b):
        with torch.no_grad():
            sim = self.sim(b.unsqueeze(0), b.unsqueeze(1))
            return b[(sim - torch.eye(*sim.shape)).argmax(0)]


class TripletLossSemiHard(torch.nn.Module):
    def negatives(self, a, b):
        with torch.no_grad():
            # Similarity between query and target
            sim = self.sim(a.unsqueeze(0), b.unsqueeze(1))
            return b[(sim - torch.eye(*sim.shape)).argmax(0)]


def build_preprocessor():
    text_field = Field(lower=True, batch_first=True)
    fields = [
        ('query', text_field),
        ('target', text_field),
    ]
    return TextPreprocessor(fields, need_vocab={
        text_field: 5})


class EmbeddingSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__embeddings=len(X.fields["query"].vocab))


def batch2dict(batch):
    return {f: attrgetter(f)(batch) for f in batch.input_fields}


class UnsupervisedNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        query, target = y_pred
        return self.criterion_(query, target)

    def predict_proba(self, X):
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            y_probas.append(cosine(*yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch2dict(batch), torch.empty(0)


def build_model():
    model = UnsupervisedNet(
        module=DSSM,
        batch_size=512,
        criterion=TripletLoss,
        iterator_train=SkorchBucketIterator,
        iterator_valid=SkorchBucketIterator,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[EmbeddingSetter()],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    train, test, val = data()
    print(train.head())


if __name__ == '__main__':
    main()
