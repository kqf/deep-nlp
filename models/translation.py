import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.translate.bleu_score import corpus_bleu

"""
!mdkir -p data
!unzip data/rus-eng.zip -d data/
!curl http://www.manythings.org/anki/rus-eng.zip -o data/rus-eng.zip
!
!pip install pandas torch torchtext sacremoses
!pip install spacy
!python -m spacy download en
"""


"""
- [x] Simple encoder-decoder architecture
- [x] Add greedy translation
- [x] Evaluate the model (BLEU)
- [ ] Beam search
- [ ] Scheduled sampling
- [ ] More layers

"""


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=3, max_tokens=16,
                 init_token="<s>", eos_token="</s>"):
        self.min_freq = min_freq
        self.max_tokens = max_tokens
        self.source_name = "source"
        self.source = Field(
            tokenize="spacy", init_token=None, eos_token=eos_token)

        self.target_name = "target"
        self.target = Field(
            tokenize="moses", init_token=init_token, eos_token=eos_token)

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
        examples = [Example.fromlist(pair, self.fields)
                    for pair in zip(sources[valid_idx], targets[valid_idx])]
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

    def forward(self, inputs, encoder_output):
        outputs, hidden = self._rnn(self._emb(inputs), encoder_output)
        return self._out(outputs), hidden


# TODO: Add the init token and the eos token as the parameters
class TranslationModel(torch.nn.Module):
    def __init__(
            self,
            source_vocab_size,
            target_vocab_size,
            emb_dim=128,
            rnn_hidden_dim=256,
            num_layers=1,
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
        return self.decoder(target_inputs, encoder_hidden)


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


def epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0

    is_train = optimizer is not None
    name = name or ""
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
                "{:>5s} Loss = {:.5f}, PPX = {:.2f}".format(
                    name, loss.item(), math.exp(loss.item())))

        bar.set_description(
            "{:>5s} Loss = {:.5f}, PPX = {:.2f}".format(
                name, epoch_loss / batches_count,
                math.exp(epoch_loss / batches_count))
        )
        bar.refresh()

    return epoch_loss / batches_count


class Translator():
    def __init__(self, mtype=TranslationModel, batch_size=32, epochs_count=5):
        self.epochs_count = epochs_count
        self.batch_size = batch_size
        self.mtype = mtype

    def fit(self, X, y=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.mtype(
            source_vocab_size=len(X.fields["source"].vocab),
            target_vocab_size=len(X.fields["target"].vocab),
        ).to(device)

        pi = X.fields["target"].vocab.stoi["<pad>"]
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pi).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        data_iter = BucketIterator(
            X,
            batch_size=self.batch_size,
            shuffle=True,
            device=device,
        )
        for i in range(self.epochs_count):
            name_prefix = "[{} / {}] ".format(i + 1, self.epochs_count)
            epoch(
                model=self.model,
                criterion=self.criterion,
                data_iter=data_iter,
                optimizer=self.optimizer,
                name=name_prefix,
            )
        print(f"Blue score: {self.score(X):.3g} %")
        return self

    def transform(self, X):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bos_index = X.fields["target"].vocab.stoi["<s>"]
        eos_index = X.fields["target"].vocab.stoi["</s>"]

        itos = X.fields["target"].vocab.itos
        outputs = []
        with torch.no_grad():
            for example in X:
                inputs = X.fields["source"].process(example.source).to(device)
                result = []

                step = torch.LongTensor([[bos_index]]).to(device)
                hidden = self.model.encoder(inputs.reshape(-1, 1))
                for _ in range(30):
                    step, hidden = self.model.decoder(step, hidden)
                    step = step.argmax(-1)

                    if step.item() == eos_index:
                        break

                    result.append(step)
                outputs.append(
                    " ".join(itos[ind.squeeze().item()] for ind in result))
        return outputs

    def score(self, data, y=None):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        refs, hyps = [], []

        bos_index = data.fields['target'].vocab.stoi["<s>"]
        eos_index = data.fields['target'].vocab.stoi["</s>"]

        data_iter = BucketIterator(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            device=device,
        )
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                encoder_hidden = self.model.encoder(batch.source)
                hidden = encoder_hidden
                # if self.model._bidirectional:
                #     hidden = None
                result = [torch.LongTensor([bos_index]).expand(
                    1, batch.target.shape[1])]

                for _ in range(30):
                    step, hidden = self.model.decoder(
                        result[-1], hidden)
                    step = step.argmax(-1)
                    result.append(step)

                targets = batch.target.data.cpu().numpy().T
                _, eos_indices = np.where(targets == eos_index)

                targets = [target[:eos_ind]
                           for eos_ind, target in zip(eos_indices, targets)]
                refs.extend(targets)

                result = torch.cat(result)
                result = result.data.cpu().numpy().T
                _, eos_indices = np.where(result == eos_index)

                result = [res[:eos_ind]
                          for eos_ind, res in zip(eos_indices, result)]
                hyps.extend(result)

        return corpus_bleu([[ref] for ref in refs], hyps) * 100


def build_model(**kwargs):
    steps = make_pipeline(
        TextPreprocessor(),
        Translator(**kwargs),
    )
    return steps


def main():
    df = data()
    model = build_model()
    model.fit(df, None)
    subsample = df.sample(10)
    subsample["translation"] = model.transform(subsample)
    print(subsample[["source", "target", "translation"]])


if __name__ == "__main__":
    main()
