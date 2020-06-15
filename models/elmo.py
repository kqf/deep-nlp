import torch

from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Field, Example, Dataset, BucketIterator


def read_dataset(path):
    data = []
    with open(path) as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line and words:
                data.append((words, tags))
                words, tags = [], []
                continue
            word, pos_tag, synt_tag, ner_tag = line.split()
            words.append(word)
            tags.append(ner_tag)
        if words:
            data.append((words, tags))
    return data[1:]


def data(dataset="train"):
    return read_dataset(f"data/conll_2003/{dataset}.txt")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            field.build_vocab(dataset)
        return self

    def transform(self, X, y=None):
        examples = [Example.fromlist(f, self.fields) for f in X]
        dataset = Dataset(examples, self.fields)
        return dataset


def build_preprocessor():
    fields = [
        ('tokens', Field(unk_token=None, batch_first=True)),
        ('tags', Field(unk_token=None, batch_first=True, is_target=True)),
    ]
    return TextPreprocessor(fields)


class BaselineTagger(torch.nn.Module):
    def __init__(self, embeddings, tags_count,
                 emb_dim=100, rnn_dim=256, num_layers=1):
        super().__init__()
        self._emb = torch.nn.Embedding.from_pretrained(embeddings)
        self._rnn = torch.nn.LSTM(
            emb_dim, rnn_dim, num_layers=num_layers, batch_first=True)
        self._out = torch.nn.Linear(rnn_dim, tags_count)

    def forward(self, inputs):
        emb = self._emb(inputs)
        hid, _ = self._rnn(emb)
        return self._out(hid)


def main():
    df = data()
    print(df[:3])
    build_preprocessor().fit_transform(df)


if __name__ == '__main__':
    main()
