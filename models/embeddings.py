import torch

from operator import attrgetter
from torchtext.data import Dataset, Example
from torchtext.data import Field, BucketIterator

from sklearn.base import BaseEstimator, TransformerMixin


def split(sentence):
    return sentence.split()


class SkipGramDataset(Dataset):

    def __init__(self, lines, fields, tokenize=split, window_size=3, **kwargs):
        examples = []
        ws = window_size
        for line in lines:
            words = tokenize(line.strip())
            if len(words) < window_size + 1:
                continue

            for i in range(len(words)):
                contexts = words[max(0, i - ws):i]
                contexts += words[
                    min(i + 1, len(words)):
                    min(len(words), i + ws) + 1
                ]

                for context in contexts:
                    examples.append(Example.fromlist(
                        (context, words[i]), fields))
        super(SkipGramDataset, self).__init__(examples, fields, **kwargs)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, dtype=SkipGramDataset):
        self.fields = fields
        self.dtype = dtype

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            field.build_vocab(dataset)
        return self

    def transform(self, X, y=None):
        return self.dtype(X, self.fields)


def build_preprocessor():
    word = Field(tokenize=lambda x: [x], batch_first=True)
    fields = [
        ('context', word),
        ('target', word)
    ]
    return TextPreprocessor(fields, dtype=SkipGramDataset)


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield self.batch2dict(batch), batch.target

    @staticmethod
    def batch2dict(batch):
        return {f: attrgetter(f)(batch) for f in batch.fields}


class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=100):
        super().__init__()
        if vocab_size is not None:
            self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        latent = self.embeddings(inputs)
        return self.out_layer(latent)


def main():
    raw = [
        "first sentence",
        "second sentence",
    ]
    data = build_preprocessor().fit_transform(raw)
    print(data)


if __name__ == '__main__':
    main()
