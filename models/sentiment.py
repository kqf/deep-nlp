import torch
import skorch
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.data import BucketIterator
from skorch.toy import MLPModule


def data():
    fname = "data/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    return pd.read_csv(fname)


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


def build_preprocessor():
    fields = [
        ("review", Field()),
        ("sentiment", LabelField(is_target=True)),
    ]
    return TextPreprocessor(fields)


def build_model():
    model = skorch.NeuralNet(
        module=MLPModule,
        module__input_units=32,
        # module__vocab_size=1,  # Dummy dimension
        # module__n_sentiments=1,
        optimizer=torch.optim.Adam,
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

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    df = data()
    train, test = train_test_split(df, test_size=0.5)

    print(train.head())
    print(train.info())


if __name__ == '__main__':
    main()
