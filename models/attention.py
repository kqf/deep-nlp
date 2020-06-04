import torch
import skorch
import random
import numpy as np
import pandas as pd

from operator import attrgetter
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


def build_preprocessor(init_token="<s>", eos_token="</s>"):
    source_field = Field(
        tokenize="spacy",
        init_token=None,
        eos_token=eos_token,
        lower=True,
        batch_first=True
    )
    target_field = Field(
        tokenize="moses",
        init_token=init_token,
        eos_token=eos_token,
        lower=True,
        batch_first=True,
        is_target=True
    )
    fields = [
        ('source', source_field),
        ('target', target_field),
    ]
    return TextPreprocessor(fields, min_freq=3)


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


class LanguageModelNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_pred, _ = y_pred
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, shift(y_true, by=1).view(-1))


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield self.batch2dict(batch), batch.target

    @staticmethod
    def batch2dict(batch):
        return {f: attrgetter(f)(batch) for f in batch.fields}


class InputVocabSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        tvocab = X.fields["target"].vocab
        svocab = X.fields["source"].vocab
        net.set_params(module__source_vocab_size=len(svocab))
        net.set_params(module__target_vocab_size=len(tvocab))
        net.set_params(criterion__ignore_index=tvocab["<pad>"])


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


class TranslationModel(torch.nn.Module):
    def __init__(
            self,
            source_vocab_size=0,
            target_vocab_size=0,
            emb_dim=128,
            rnn_hidden_dim=256,
            num_layers=1,
            bidirectional_encoder=False,
            encodertype=Encoder,
            decodertype=Decoder,
    ):

        super().__init__()

        self.encoder = encodertype(
            source_vocab_size, emb_dim,
            rnn_hidden_dim, num_layers, bidirectional_encoder)

        self.decoder = decodertype(
            target_vocab_size, emb_dim,
            rnn_hidden_dim, num_layers)

    def forward(self, source, target):
        # Convert to batch second
        sources, targets = source.T, target.T

        # Run the model
        encoded, hidden = self.encoder(sources)
        return self.decoder(targets, encoded, hidden)


def build_model():
    model = LanguageModelNet(
        module=TranslationModel,
        criterion=torch.nn.CrossEntropyLoss,
        batch_size=512,
        iterator_train=SkorchBucketIterator,
        iterator_valid=SkorchBucketIterator,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            InputVocabSetter(),
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
    print(df.columns)


if __name__ == '__main__':
    main()
