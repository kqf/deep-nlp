import torch
import pandas as pd
from torchtext.data import Field, Example, Dataset

"""
!curl http://www.manythings.org/anki/rus-eng.zip -o data/rus-eng.zip
!
!pip install pandas torch, torchtext
!pip install spacy
!python -m spacy download en
"""


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


class TextPreprocessor:
    def __init__(self, min_freq=3, corpus_fraction=0.3, max_tokens=16,
                 init_token="<s>", eos_token="</s>"):
        self.min_freq = min_freq
        self.corpus_fraction = corpus_fraction
        self.max_tokens = max_tokens
        self.source_name = "source"
        self.source = Field(
            tokenize='spacy', init_token=None, eos_token=eos_token)

        self.target_name = "target"
        self.target = Field(
            tokenize='moses', init_token=init_token, eos_token=eos_token)

        self.fields = [
            (self.source_name, self.source),
            (self.target_name, self.target),
        ]

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        self.source.build_vocab(dataset, min_freq=self.min_freq)
        self.target.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        sources = X[self.source_name].apply(self.source.preprocess)
        targets = X[self.target_name].apply(self.target.preprocess)
        valid_idx = (
            (sources.str.len() < self.max_tokens) & (
                targets.str.len() < self.max_tokens)
        )
        out = pd.DataFrame([sources[valid_idx], targets[valid_idx]])
        out = out.sample(frac=self.corpus_fraction)
        examples = [Example.fromlist(pair, self.fields)
                    for pair in out.values]
        dataset = Dataset(examples, self.fields)
        return dataset


def main():
    df = data()
    print(df.head())
    print(TextPreprocessor().fit(df))


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=128, rnn_hidden_dim=256,
                 num_layers=1, bidirectional=False):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        return self._rnn(self._emb(inputs))[0]


if __name__ == '__main__':
    main()
