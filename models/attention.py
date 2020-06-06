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
        batch_first=True
    )
    target_field = Field(
        tokenize="moses",
        init_token=init_token,
        eos_token=eos_token,
        batch_first=True,
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
        return self.criterion_(logits, shift(y_true.T, by=1).view(-1))


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

    def forward(self, inputs, encoder_output, hidden=None):
        outputs, hidden = self._rnn(self._emb(inputs), hidden)
        return self._out(outputs), hidden


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self._gamma = torch.nn.Parameter(torch.ones(features))
        self._beta = torch.nn.Parameter(torch.zeros(features))
        self._eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdims=True)
        var = inputs.std(dim=-1, keepdims=True) ** 2
        inputs = (inputs - mean) / (var + self._eps).sqrt()
        return self._gamma * inputs + self._beta


class ResidualBlock(torch.nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        return inputs + self._dropout(sublayer(self._norm(inputs)))


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self._dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        f_att = torch.matmul(query, key.transpose(-2, -1))
        f_att /= np.sqrt(key.shape[-1])
        f_att = torch.masked_fill_(mask.unsqueeze(-2) == 0, -float('inf'))
        weights = torch.funtional.softmax(f_att, -1)
        output = torch.matmul(weights, value)
        return self._dropout(output), weights


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        return self.w_2(self.dropout(torch.functional.relu(self.w_1(inputs))))


class EncoderBlock(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, mask):
        outputs = self._self_attention_block(
            inputs,
            lambda inputs: self._self_attn(inputs, inputs, inputs, mask)
        )
        return self._feed_forward_block(outputs, self._feed_forward)


class DecoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, encoder_attn,
                 feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._encoder_attn = encoder_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        outputs = self._self_attention_block(
            inputs, lambda inputs: self._self_attn(
                inputs, inputs, inputs, target_mask)
        )
        outputs = self._attention_block(
            outputs, lambda inputs: self._encoder_attn(
                inputs, encoder_output, encoder_output, source_mask)
        )
        return self._feed_forward_block(outputs, self._feed_forward)


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


def ppx(loss_type):
    def _ppx(model, X, y):
        return np.exp(model.history[-1][loss_type])
    _ppx.__name__ = f"ppx_{loss_type}"
    return _ppx


def build_model():
    model = LanguageModelNet(
        module=TranslationModel,
        optimizer=torch.optim.Adam,  # <<< unexpected
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=SkorchBucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SkorchBucketIterator,
        iterator_valid__shuffle=True,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            InputVocabSetter(),
            skorch.callbacks.GradientNormClipping(1.),
            skorch.callbacks.EpochScoring(ppx("train_loss"), on_train=True),
            skorch.callbacks.EpochScoring(ppx("valid_loss"), on_train=False),
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