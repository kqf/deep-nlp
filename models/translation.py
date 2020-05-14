import io
import math
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial
from torchtext.data import Field, Example, Dataset, BucketIterator
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""
!mkdir -p data/
!curl http://www.manythings.org/anki/rus-eng.zip -o data/rus-eng.zip
!!unzip data/rus-eng.zip -d data/
!pip install pandas torch torchtext sacremoses
!pip install spacy
!python -m spacy download en
"""


"""
- [x] Simple encoder-decoder architecture
- [x] Add greedy translation
- [x] Evaluate the model (BLEU)
- [x] Beam search
- [X] Scheduled sampling?
- [ ] Dropout
- [ ] More layers
- [X] Byte-pair decoding
- [ ] Attention
    - [x] Additive
    - [ ] Dot attention
    - [ ] Multiplicative

"""


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


def fit_bpe(data, num_symbols):
    outfile = io.StringIO()
    learn_bpe(data, outfile, num_symbols)
    return BPE(outfile)


class SubwordTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=["source", "target"], out="bpe", num_symbols=3000):
        self.num_symbols = num_symbols
        self.out = out
        self.cols = cols
        self._rules = {}

    def fit(self, X, y=None):
        for c in self.cols:
            # Since the dataset is small -- do this in memory
            self._rules[c] = fit_bpe(X[c].values, self.num_symbols)
        return self

    def transform(self, X, y=None):
        for c in self.cols:
            X[f"{c}_{self.out}"] = self._rules[c]
        return X


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=3, max_tokens=16, bpe_col_prefix=None,
                 init_token="<s>", eos_token="</s>"):
        self.bpe_col_prefix = bpe_col_prefix
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

        if self.bpe_col_prefix is not None:
            source_bpe = X[f"{self.source_name}_{self.bpe_col_prefix}"].iloc[0]
            sources = sources.apply(source_bpe.segment_tokens)

            target_bpe = X[f"{self.target_name}_{self.bpe_col_prefix}"].iloc[0]
            targets = targets.apply(target_bpe.segment_tokens)

        valid_idx = (
            (sources.str.len() < self.max_tokens) & (
                targets.str.len() < self.max_tokens)
        )
        examples = [Example.fromlist(pair, self.fields)
                    for pair in zip(sources[valid_idx], targets[valid_idx])]
        dataset = Dataset(examples, self.fields)
        return dataset


class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_size, key_size, hidden_dim):
        super().__init__()

        self._query_layer = torch.nn.Linear(query_size, hidden_dim)
        self._key_layer = torch.nn.Linear(key_size, hidden_dim)
        self._energy_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value, mask):
        """
        query: FloatTensor with shape (batch_size, query_size) (h_i)
        key: FloatTensor with shape (encoder_seq_len, batch_size, key_size) (sequence of s_1, ..., s_m)
        value: FloatTensor with shape (encoder_seq_len, batch_size, key_size) (sequence of s_1, ..., s_m)
        mask: ByteTensor with shape (encoder_seq_len, batch_size) (ones in positions of <pad> tokens, zeros everywhere else)
        """  # noqa

        # tanh = [encoder_seq_len, batch_size, hidden_dim]
        tanh = torch.tanh(self._query_layer(query) + self._key_layer(key))

        # f_att = [encoder_seq_len, batch_size, 1]
        f_att = self._energy_layer(tanh)

        # Mask out pads: after softmax the masked weights will be 0
        f_att.data.masked_fill_(mask.unsqueeze(2), -float('inf'))

        # softmax-normalized f_att weight
        weights = F.softmax(f_att, 0)

        # find the context vector as a weighed sum of value (s_1, ..., s_m)
        return (weights * value).sum(0), weights


class DotAttention(torch.nn.Module):
    def __init__(self, query_size, key_size, hidden_dim):
        super().__init__()

    def forward(self, query, key, value, mask):
        # assume Q = K
        # ([B, Q] -> [B, Q, 1]) * ([T, B, K] -> [B, K, T]) = [B, 1, T]
        f_att_s = torch.bmm(query.unsqueeze(1), key.permute(1, 2, 0))
        f_att = f_att_s.transpose(-1, -2)

        # [B, T, 1] -> [T, B, 1]
        f_att = f_att.transpose(0, 1)
        # import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa

        f_att.data.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        weights = F.softmax(f_att, -1)
        return (weights * value).sum(0), weights


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


class ScheduledSamplingDecoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=128,
                 rnn_hidden_dim=256, num_layers=1, sampling_rate=0.5):
        super().__init__()

        # self.p = sampling_rate
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers
        )
        self._out = torch.nn.Linear(rnn_hidden_dim, vocab_size)

    def forward(self, inputs, encoder_output, encoder_mask, hidden=None):
        embeddings = inputs
        step = inputs[0]
        result = []
        for original in embeddings:
            output, hidden = self._rnn(self._emb(step).unsqueeze(0), hidden)
            result.append(output)

            step = original
            if self.training and bool(np.random.binomial(n=1, p=0.5)):
                with torch.no_grad():
                    step = self._out(output.detach()).argmax(-1).squeeze(0)

        outputs = torch.cat(result)
        return self._out(outputs), hidden


class AttentionDecoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=128,
                 rnn_hidden_dim=256, num_layers=1, atype=AdditiveAttention):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._attention = atype(rnn_hidden_dim, rnn_hidden_dim, rnn_hidden_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim + rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers
        )
        self._out = torch.nn.Linear(rnn_hidden_dim, vocab_size)

    def logits_with_attention(self, inputs, encoder_output,
                              encoder_mask, hidden=None):
        embeddings = self._emb(inputs)
        outputs, attentions = [], []
        for emb in embeddings:
            context, weights = self._attention(
                hidden, key=encoder_output, value=encoder_output,
                mask=encoder_mask)

            rnn_input = torch.cat([emb.unsqueeze(0), context.unsqueeze(0)], -1)
            out, hidden = self._rnn(rnn_input, hidden)

            outputs.append(out)
            attentions.append(weights)

        outputs = torch.cat(outputs)
        return self._out(outputs), hidden, weights

    def forward(self, inputs, encoder_output, encoder_mask, hidden=None):
        outputs, hidden, _ = self.logits_with_attention(
            inputs, encoder_output, encoder_mask, hidden)
        return outputs, hidden


# TODO: Add the init token and the eos token as the parameters
class TranslationModel(torch.nn.Module):
    def __init__(
            self,
            source_vocab_size,
            target_vocab_size,
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

    def forward(self, source_inputs, target_inputs):
        encoder_mask = (source_inputs == 1.)  # find mask for padding inputs
        output, hidden = self.encoder(source_inputs)
        return self.decoder(target_inputs, output, encoder_mask, hidden)


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
    def __init__(self, mtype=TranslationModel, batch_size=32, epochs_count=8):
        self.epochs_count = epochs_count
        self.batch_size = batch_size
        self.mtype = mtype
        self.n_beams = None
        self.model = None

    def model_init(self, source_vocab_size, target_vocab_size):
        if self.model is None:
            self.model = self.mtype(
                source_vocab_size=source_vocab_size,
                target_vocab_size=target_vocab_size
            )
        return self.model

    def fit(self, X, y=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model_init(
            source_vocab_size=len(X.fields["source"].vocab),
            target_vocab_size=len(X.fields["target"].vocab),
        ).to(device)

        pi = X.fields["target"].vocab.stoi["<pad>"]
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pi).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        train_dataset, test_dataset = X.split(split_ratio=0.7)

        data_iter, test_iter = BucketIterator.splits(
            (train_dataset, test_dataset),
            batch_sizes=(self.batch_size, self.batch_size * 4),
            shuffle=True,
            device=device,
            sort=False,
        )
        for i in range(self.epochs_count):
            name_prefix = "[{} / {}] ".format(i + 1, self.epochs_count)
            epoch(
                model=self.model,
                criterion=self.criterion,
                data_iter=data_iter,
                optimizer=self.optimizer,
                name=f"Train {name_prefix}",
            )
            epoch(
                model=self.model,
                criterion=self.criterion,
                data_iter=test_iter,
                name=f"Valid {name_prefix}",
            )
            print(f"Blue score: {self.score(test_dataset):.3g} %")
        return self

    def score(self, data, y=None):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        refs, hyps = [], []

        bos_index = data.fields["target"].vocab.stoi["<s>"]
        eos_index = data.fields["target"].vocab.stoi["</s>"]

        data_iter = BucketIterator(
            data,
            batch_size=self.batch_size,
            shuffle=True,
            device=device,
        )
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                encoded, hidden = self.model.encoder(batch.source)
                encoder_mask = (batch.source == 1)

                result = [torch.LongTensor([bos_index]).expand(
                    1, batch.target.shape[1]).to(device)]

                for _ in range(30):
                    step, hidden = self.model.decoder(
                        result[-1], encoded, encoder_mask, hidden)
                    step = step.argmax(-1)
                    result.append(step)

                targets = batch.target.data.cpu().numpy().T
                eos_indices = (targets == eos_index).argmax(-1)
                eos_indices[eos_indices == 0] = targets.shape[1]

                targets = [target[:eos_ind]
                           for eos_ind, target in zip(eos_indices, targets)]
                refs.extend(targets)

                result = torch.cat(result).data.cpu().numpy().T
                eos_indices = (result == eos_index).argmax(-1)
                eos_indices[eos_indices == 0] = result.shape[1]

                result = [res[:eos_ind]
                          for eos_ind, res in zip(eos_indices, result)]
                hyps.extend(result)

        return corpus_bleu([[ref] for ref in refs], hyps) * 100

    def transform(self, X):
        if self.n_beams is not None:
            return self._beam_search_decode(X)
        return self._greedy_decode(X)

    def _greedy_decode(self, X):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bos_index = X.fields["target"].vocab.stoi["<s>"]
        eos_index = X.fields["target"].vocab.stoi["</s>"]

        itos = X.fields["target"].vocab.itos
        outputs = []
        with torch.no_grad():
            for example in X:
                inputs = X.fields["source"].process(
                    [example.source]).to(device)
                encoded, hidden = self.model.encoder(inputs)

                step = torch.LongTensor([[bos_index]]).to(device)
                result = []
                encoder_mask = (inputs == 1)
                for _ in range(30):
                    step, hidden = self.model.decoder(
                        step, encoded, encoder_mask, hidden)
                    step = step.argmax(-1)

                    if step.item() == eos_index:
                        break

                    result.append(step)
                outputs.append(
                    " ".join(itos[ind.squeeze().item()] for ind in result))
        return outputs

    def _beam_search_decode(self, X, beam_size=5):
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bos_index = X.fields["target"].vocab.stoi["<s>"]
        eos_index = X.fields["target"].vocab.stoi["</s>"]

        with torch.no_grad():
            for example in X:
                inputs = X.fields["source"].process(
                    [example.source]).to(device)
                encoder_hidden = self.model.encoder(inputs)
                hidden = encoder_hidden
                # if self.model._bidirectional:
                #     hidden = None
                beams = [([bos_index], 0, hidden)]

                for _ in range(30):
                    _beams = []
                    for beam in beams:
                        inputs = torch.LongTensor([[beam[0][-1]]]).to(device)
                        step, hidden = self.model.decoder(inputs, hidden)
                        step = F.log_softmax(step, -1)
                        positions = torch.topk(step, beam_size, dim=-1)[1]

                        for i in range(beam_size):
                            seq = beam[0] + [positions[0, 0, i].item()]
                            score = beam[1] - step[0, 0, positions[0, 0, i]].item()  # noqa
                            _beams.append((seq, score, hidden))

                    beams = sorted(_beams, key=lambda x: x[1])[:beam_size]

                result = sorted(beams, key=lambda x: x[1])[0][0][1:]

                end_index = np.where(np.array(result) == eos_index)

        final = result[:end_index[0][0]]
        return " ".join(X.fields["target"].vocab.itos[ind] for ind in final)


def build_model(**kwargs):
    text = make_pipeline(
        TextPreprocessor(),
    )

    steps = make_pipeline(
        text,
        Translator(**kwargs),
    )
    return steps


def build_model_bpe(**kwargs):
    text = make_pipeline(
        SubwordTransformer(),
        TextPreprocessor(bpe_col_prefix="bpe"),
    )
    # Keep the two-level pipeline for uniform tests
    steps = make_pipeline(
        text,
        Translator(**kwargs),
    )
    return steps


def validate(title, model, train, test, sample):
    print(title)
    sample = pd.DataFrame(sample)
    model.fit(train, None)
    print(f"Test set BLEU {model.score(test)} %")

    sample["translation"] = model.transform(sample)
    print(sample[["source", "translation"]])
    print("\n--------------------\n")


def main():
    df = data()
    train, test = train_test_split(df)
    sample = test.sample(10)

    model = build_model()
    validate("Basic encoder-decoder", model, train, test, sample)

    stype = partial(TranslationModel, decodertype=ScheduledSamplingDecoder)
    smodel = build_model(mtype=stype)
    validate("Scheduled sampling", smodel, train, test, sample)

    bpe_model = build_model_bpe()
    validate("BPE encoding", bpe_model, train, test, sample)

    stype = partial(TranslationModel, decodertype=AttentionDecoder)
    aamodel = build_model(mtype=stype)
    validate("Additive attention", aamodel, train, test, sample)


if __name__ == "__main__":
    main()
