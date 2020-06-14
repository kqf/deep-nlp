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


def shift(seq, by, batch_dim=0):
    padding = seq.new_ones(seq.shape[batch_dim], by)
    return torch.cat((seq[:, by:], padding), dim=-1)


class LanguageModelNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, shift(y_true, by=1).view(-1))

    def transform(self, X, max_len=10):
        self.module_.eval()
        dataset = self.get_dataset(X)
        tg = X.fields["target"]
        init_token_idx = tg.vocab.stoi[tg.init_token]
        predicted_sentences = []
        for (data, _) in self.get_iterator(dataset, training=False):
            source = data["source"]
            source_mask = self.module_.source_mask(source)
            with torch.no_grad():
                enc_src = self.module_.encoder(source, source_mask)

            target = source.new_ones(source.shape[0], 1) * init_token_idx
            for i in range(max_len + 1):
                target_mask = self.module_.target_mask(target)
                with torch.no_grad():
                    output = self.module_.decoder(
                        target, enc_src, source_mask, target_mask)

                last_pred = output[:, [-1]]
                target = torch.cat([target, last_pred.argmax(-1)], dim=-1)

            # Ensure the sequence has an end
            sentences = target.numpy()
            sentences[:, -1] = tg.vocab.stoi[tg.eos_token]

            # Ignore start of sequence token
            for seq in sentences[:, 1:]:
                stop = np.argmax(seq == tg.vocab.stoi[tg.eos_token])
                predicted_sentences.append(
                    (" ".join(np.take(tg.vocab.itos, seq[: stop]))))

        return predicted_sentences


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield self.batch2dict(batch), batch.target

    @staticmethod
    def batch2dict(batch):
        return {f: attrgetter(f)(batch) for f in batch.fields}


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        tvocab = X.fields["target"].vocab
        svocab = X.fields["source"].vocab
        net.set_params(module__source_vocab_size=len(svocab))
        net.set_params(module__target_vocab_size=len(tvocab))
        net.set_params(module__source_pad_idx=svocab["<pad>"])
        net.set_params(module__target_pad_idx=tvocab["<pad>"])
        net.set_params(criterion__ignore_index=tvocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, n_pos):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(n_pos, d_model)
        position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self._norm = torch.nn.LayerNorm(size)
        self._dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        return self._norm(inputs + self._dropout(sublayer(inputs)))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        if hid_dim % n_heads != 0:
            raise IOError(
                "Attention hid_dim should be multiple of n_heads. "
                f"Got hid_dim={hid_dim} and n_heads={n_heads} instead."
            )

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # head_dim * n_heads = hid_dim
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # energy = [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -float('inf'))

        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim=-1)

        # x = [batch size, n heads, query len, head dim]
        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self._all = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, inputs):
        return self._all(inputs)


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        super().__init__()
        self._attn = MultiHeadedAttention(d_model, n_heads, dropout_rate)
        self._ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self._res1 = ResidualBlock(d_model, dropout_rate)
        self._res2 = ResidualBlock(d_model, dropout_rate)

    def forward(self, src, mask):
        enc = self._res1(src, lambda x: self._attn(x, x, x, mask)[0])
        return self._res2(enc, self._ff)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, d_ff,
                 n_blocks, n_heads, dropout_rate, n_pos=128):
        super().__init__()
        self._emb = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, d_model),
            PositionalEncoding(d_model, dropout_rate, n_pos),
        )

        self._blocks = torch.nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_blocks)
        ])
        self._norm = torch.nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
        inputs = self._emb(inputs)
        for block in self._blocks:
            inputs = block(inputs, mask)
        return self._norm(inputs)


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate):
        super().__init__()

        self._attn = MultiHeadedAttention(d_model, n_heads, dropout_rate)
        self._attn_enc = MultiHeadedAttention(d_model, n_heads, dropout_rate)
        self._ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self._res1 = ResidualBlock(d_model, dropout_rate)
        self._res2 = ResidualBlock(d_model, dropout_rate)
        self._res3 = ResidualBlock(d_model, dropout_rate)

    def forward(self, x, enc_src, source_mask, target_mask):
        x = self._res1(x, lambda x: self._attn(x, x, x, target_mask)[0])
        e = enc_src
        x = self._res2(x, lambda x: self._attn_enc(x, e, e, source_mask)[0])
        return self._res3(x, self._ff)


class Decoder(torch.nn.Module):
    def __init__(self,
                 vocab_size, d_model, d_ff, n_blocks,
                 n_heads, dropout_rate, n_pos=128):
        super().__init__()

        self._emb = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, d_model),
            PositionalEncoding(d_model, dropout_rate, n_pos),
        )

        self._blocks = torch.nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_blocks)])

        self._norm = torch.nn.LayerNorm(d_model)
        self._out_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(self, inputs, enc_src, source_mask, target_mask):
        inputs = self._emb(inputs)
        for block in self._blocks:
            inputs = block(inputs, enc_src, source_mask, target_mask)
        return self._out_layer(self._norm(inputs))


class TranslationModel(torch.nn.Module):
    def __init__(
            self,
            source_vocab_size=0,
            target_vocab_size=0,
            source_pad_idx=-1,
            target_pad_idx=-1,
            d_model=256,
            d_ff=1024,
            blocks_count=4,
            heads_count=8,
            dropout_rate=0.1):
        super().__init__()

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

        self.d_model = d_model
        self.encoder = Encoder(source_vocab_size, d_model,
                               d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(target_vocab_size, d_model,
                               d_ff, blocks_count, heads_count, dropout_rate)

    def forward(self, source, target):
        source_mask = self.source_mask(source)
        target_mask = self.target_mask(target)

        enc_src = self.encoder(source, source_mask)
        return self.decoder(target, enc_src, source_mask, target_mask)

    def source_mask(self, inputs):
        # mask = [batch_size, 1, 1, seq_len]
        return (inputs != self.source_pad_idx).unsqueeze(-2).unsqueeze(-2)

    def target_mask(self, inputs):
        # mask = [batch_size, 1, 1, seq_len]
        mask = (inputs != self.target_pad_idx).unsqueeze(-2).unsqueeze(-2)

        tlen = inputs.shape[-1]
        # subsequent_mask = [tlen, tlen]
        subsequent_mask = torch.tril(
            torch.ones(tlen, tlen, device=inputs.device))

        # mask = [batch_size, 1, tlen, tlen]
        target_mask = mask & subsequent_mask.type_as(mask)
        return target_mask


def ppx(loss_type):
    def _ppx(model, X, y):
        return np.exp(model.history[-1][loss_type])
    _ppx.__name__ = f"ppx_{loss_type}"
    return _ppx


def initialize_weights(m):
    if m.data.dim() > 1:
        torch.nn.init.xavier_uniform_(m.data)


class NoamOpt(torch.optim.Adam):
    def __init__(self, params,
                 d_model=1, factor=2, warmup=4000, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor

        self._rate = 0
        self._step = 0

    def step(self, step_fn):
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        return super().step(step_fn)

    def rate(self, step=None):
        if step is None:
            step = self._step
        factor = self.d_model ** (-0.5) * self.factor
        return factor * min(step ** (-0.5), step * self.warmup ** (-1.5))


def build_model():
    model = LanguageModelNet(
        module=TranslationModel,
        module__d_model=256,
        module__d_ff=1024,
        module__blocks_count=4,
        module__heads_count=8,
        module__dropout_rate=0.1,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0005,
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
            DynamicVariablesSetter(),
            skorch.callbacks.GradientNormClipping(1.),
            skorch.callbacks.EpochScoring(ppx("train_loss"), on_train=True),
            skorch.callbacks.EpochScoring(ppx("valid_loss"), on_train=False),
            skorch.callbacks.Initializer('*', fn=initialize_weights),
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
