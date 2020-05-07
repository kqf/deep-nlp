import math
import torch
import pandas as pd
from tqdm import tqdm
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
        return self._rnn(self._emb(inputs))[-1]


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
        outputs, hidden = self._rnn(self._emb(inputs), encoder_output)
        return self._out(outputs), hidden


class TranslationModel(torch.nn.Module):
    def __init__(self,
                 source_vocab_size, target_vocab_size, emb_dim=128,
                 rnn_hidden_dim=256, num_layers=1,
                 bidirectional_encoder=False):

        super().__init__()

        self.encoder = Encoder(
            source_vocab_size, emb_dim,
            rnn_hidden_dim, num_layers, bidirectional_encoder)

        self.decoder = Decoder(
            target_vocab_size, emb_dim,
            rnn_hidden_dim, num_layers)

    def forward(self, source_inputs, target_inputs):
        encoder_hidden = self.encoder(source_inputs)
        return self.decoder(target_inputs, encoder_hidden, encoder_hidden)


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


def epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0

    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        bar = tqdm(enumerate(data_iter), total=batches_count)
        for i, batch in bar:
            logits, _ = model(batch.source, batch.target)

            # [target_seq_size, batch] -> [target_seq_size, batch]
            target = shift(batch.target, by=1)

            loss = criterion(
                # [target_seq_size * batch, target_vocab_size]
                logits.view(-1, logits.shape[-1]),
                # [target_seq_size * batch]
                target.view(-1)
            )

            epoch_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()

            bar.update()
            bar.set_description(
                '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                    name, loss.item(), math.exp(loss.item())))

        bar.set_description(
            '{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, epoch_loss / batches_count,
                math.exp(epoch_loss / batches_count))
        )
        bar.refresh()

    return epoch_loss / batches_count


def main():
    df = data()
    print(df.head())
    print(TextPreprocessor().fit(df))


if __name__ == '__main__':
    main()
