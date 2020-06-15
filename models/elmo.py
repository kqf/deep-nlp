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


def main():
    df = data()
    print(df[:3])
    build_preprocessor().fit_transform(df)


if __name__ == '__main__':
    main()
