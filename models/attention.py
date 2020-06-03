import torch
import skorch
import random
import numpy as np
import pandas as pd
from skorch.toy import MLPModule
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=0):
        self.fields = fields
        self.min_freq = min_freq or {}

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        dataset = Dataset(examples, self.fields)
        return dataset


def build_preprocessor():
    source_field = Field(lower=True, batch_first=True)
    target_field = Field(lower=True, batch_first=True)
    fields = [
        ('source', source_field),
        ('target', target_field),
    ]
    return TextPreprocessor(fields, min_freq=5)


class LanguageModelNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_shape = (y_pred.shape[0],)
        return self.criterion_(y_pred, torch.randint(0, 2, y_shape))


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.source[:, :2].float(), torch.empty(0)


class InputVocabSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        pi = X.fields["target"].vocab["<pad>"]
        net.set_params(criterion__ignore_index=pi)


def build_model():
    model = LanguageModelNet(
        module=MLPModule,
        module__input_units=2,
        criterion=torch.nn.CrossEntropyLoss,
        batch_size=512,
        iterator_train=SkorchBucketIterator,
        iterator_valid=SkorchBucketIterator,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[InputVocabSetter()],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


def main():
    df = data()
    print(df.columns)


if __name__ == '__main__':
    main()
