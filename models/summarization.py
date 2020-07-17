import torch
import skorch
import random
import numpy as np
import pandas as pd
from operator import attrgetter

from sklearn.pipeline import make_pipeline
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.base import BaseEstimator, TransformerMixin


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
!mkdir -p data/
!curl -k -L "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab" -o data/news.zip
!pip install torch
!pip install torchtext
!pip install sacremoses
"""  # noqa

"""
    - [x] Boilerplate
    - [x] Decoder encoder
    - [ ] Transformer
"""


def data():
    raw = pd.read_csv("data/news.zip")
    df = raw.rename(columns={"text": "source", "title": "target"})
    return df[["source", "target"]]


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


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


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


class SummarizationModel(torch.nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_dim=128,
            rnn_hidden_dim=256,
            num_layers=1,
            bidirectional_encoder=False,
            encodertype=Encoder,
            decodertype=Decoder,
    ):

        super().__init__()

        self.encoder = encodertype(
            vocab_size, emb_dim,
            rnn_hidden_dim, num_layers, bidirectional_encoder)

        self.decoder = decodertype(
            vocab_size, emb_dim,
            rnn_hidden_dim, num_layers)

    def forward(self, source, target):
        source, target = source.T, target.T
        encoder_mask = (source == 1.)  # find mask for padding inputs
        output, hidden = self.encoder(source)
        return self.decoder(target, output, encoder_mask, hidden)


class LanguageModelNet(skorch.NeuralNet):
    n_beams = None

    def get_loss(self, y_pred, y_true, X=None, training=False):
        out, _ = y_pred
        logits = out.view(-1, out.shape[-1])
        return self.criterion_(logits, shift(y_true.T, by=1).view(-1))

    def _decode_iterator(self, X, max_len):
        if self.n_beams is None:
            return self._greedy_decode_iterator(X, max_len)
        return self._beam_decode_iterator(X, max_len, self.n_beams)

    def _greedy_decode_iterator(self, X, max_len=100):
        self.module_.eval()
        dataset = self.get_dataset(X)
        tg = X.fields["target"]
        init_token_idx = tg.vocab.stoi[tg.init_token]
        for (data, _) in self.get_iterator(dataset, training=False):
            source = data["source"].T
            source_mask = (source == 1)
            with torch.no_grad():
                enc_src, hidden = self.module_.encoder(source)

            target = source.new_ones(source.shape[1], 1) * init_token_idx
            for i in range(max_len + 1):
                with torch.no_grad():
                    output, hidden = self.module_.decoder(
                        target.T, enc_src, source_mask, hidden)

                last_pred = output[[-1], :]
                target = torch.cat([target, last_pred.argmax(-1).T], dim=-1)

            # Ensure the sequence has an end
            sentences = target.numpy()
            sentences[:, -1] = tg.vocab.stoi[tg.eos_token]
            yield data, sentences

    def transform(self, X, max_len=10):
        tg = X.fields["target"]
        pred = []
        for X, sentences in self._decode_iterator(X, max_len):
            for seq in sentences[:, 1:]:
                stop = np.argmax(seq == tg.vocab.stoi[tg.eos_token])
                pred.append((" ".join(np.take(tg.vocab.itos, seq[: stop]))))
        return pred


def build_preprocessor(init_token="<s>", eos_token="</s>"):
    text = Field(
        batch_first=True,
        tokenize="spacy",
        init_token=init_token,
        eos_token=eos_token,
    )

    fields = [
        ("source", text),
        ("target", text),
    ]
    return TextPreprocessor(fields)


class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield self.batch2dict(batch), batch.target

    @staticmethod
    def batch2dict(batch):
        return {f: attrgetter(f)(batch) for f in batch.fields}


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["source"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(module=SummarizationModel, packed=False, bidirectional=False):
    model = LanguageModelNet(
        module=module,
        module__vocab_size=1000,  # Dummy dimension
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        batch_size=32,
        iterator_train=SkorchBucketIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SkorchBucketIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=lambda x, y, **kwargs: Dataset.split(x, **kwargs),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
            DynamicVariablesSetter(),
        ],
    )

    full = make_pipeline(
        build_preprocessor(packed),
        model,
    )
    return full


def main():
    df = data()
    print(df.head())
    model = build_model()
    model.fit(df, None)

    samples = df.sample(10)
    samples["generated"] = model.transform(samples)
    print(samples["source", "target", "generated"])


if __name__ == '__main__':
    main()
