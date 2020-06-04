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
    target_field = Field(lower=True, batch_first=True, is_target=True)
    fields = [
        ('source', source_field),
        ('target', target_field),
    ]
    return TextPreprocessor(fields, min_freq=5)


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


class LanguageModelNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        # Next line is just a hack to match the shapes
        y_pred = torch.cat([y_pred.unsqueeze(1)] * y_true.shape[1], 1)

        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, shift(y_true, by=1).view(-1))


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.source[:, :2].float(), batch.target


class InputVocabSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["target"].vocab
        net.set_params(criterion__ignore_index=vocab["<pad>"])
        net.set_params(module__output_units=len(vocab))


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
        return self._rnn(self._emb(inputs))


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=128,
                 rnn_hidden_dim=256, num_layers=1):
        super().__init__()

        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers
        )
        self._out = torch.nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, inputs, encoder_output, encoder_mask, hidden=None):
        outputs, hidden = self._rnn(self._emb(inputs), hidden)
        return self._out(outputs), hidden


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
