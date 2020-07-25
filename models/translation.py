import io
import torch
import skorch
import random
import numpy as np
import pandas as pd

from collections import namedtuple
from operator import attrgetter

from functools import partial

from torchtext.data import Dataset, Example, Field
from torchtext.data import BucketIterator

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
- [x] Attention
    - [x] Additive
    - [x] Dot attention
    - [x] Multiplicative

"""


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


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


def fit_bpe(data, num_symbols):
    outfile = io.StringIO()
    learn_bpe(data, outfile, num_symbols)
    return BPE(outfile)


class SubwordPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1, num_symbols=3000):
        self.fields = fields
        self.min_freq = min_freq
        self.num_symbols = num_symbols
        self._bpe = {}

    def fit(self, X, y=None):
        # First learn the bpe
        for c, _ in self.fields:
            self._bpe[c] = fit_bpe(X[c].values, self.num_symbols)

        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [
            X[col].apply(f.preprocess).apply(self._bpe[col].segment_tokens)
            for col, f in self.fields
        ]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        return Dataset(examples, self.fields)


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
        weights = torch.nn.functional.softmax(f_att, 0)

        # find the context vector as a weighed sum of value (s_1, ..., s_m)
        return (weights * value).sum(0), weights


class DotAttention(torch.nn.Module):
    def __init__(self, query_size, key_size, hidden_dim):
        super().__init__()

    def forward(self, query, key, value, mask):
        # assume Q = K
        # ([B, Q] -> [B, Q, 1]) * ([T, B, K] -> [B, K, T]) = [B, 1, T]
        f_att_s = query.unsqueeze(1) @ key.permute(1, 2, 0)
        f_att = f_att_s.transpose(-1, -2)

        # [B, T, 1] -> [T, B, 1]
        f_att = f_att.transpose(0, 1)

        f_att.data.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        weights = torch.nn.functional.softmax(f_att, -1)
        return (weights * value).sum(0), weights


class MultiplicativeAttention(torch.nn.Module):
    def __init__(self, query_size, key_size, hidden_dim):
        super().__init__()
        self._key = torch.nn.Linear(key_size, query_size)

    def forward(self, query, key, value, mask):
        # [B, Q] -> [B, 1, Q]
        Q = query.unsqueeze(1)

        # self._key([T, B, K]) -> [T, B, Q] -> [B, Q, T]
        K = self._key(key).permute(1, 2, 0)

        # [B, 1, Q] @ [B, Q, T] -> [B, 1, T] -> [T, B, 1]
        f_att = (Q @ K).permute(2, 0, 1)
        f_att.data.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        weights = torch.nn.functional.softmax(f_att, -1)
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


class ConvEncoder(torch.nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 hid_dim=256,
                 n_layers=1,
                 kernel_size=3,
                 dropout=0.5,
                 max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.tok_embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, emb_dim)

        self.emb2hid = torch.nn.Linear(emb_dim, hid_dim)
        self.hid2emb = torch.nn.Linear(hid_dim, emb_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels=hid_dim,
                out_channels=2 * hid_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            )
            for _ in range(n_layers)])

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, src_len]
        batch_size, src_len = src.shape

        # create position tensor
        # pos = [0, 1, 2, 3, ..., src_len - 1]
        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(
            0).repeat(batch_size, 1).to(src.device)

        # embed tokens and positions
        # both[batch_size, src_len, emb_dim]
        token_embedded = self.tok_embedding(src)
        position_embedded = self.pos_embedding(pos)

        # combine embeddings by elementwise summing
        # embedded [batch_size, src_len, emb_dim]
        embedded = self.dropout(token_embedded + position_embedded)

        # pass embedded through linear layer to convert from emb_dim to hid_dim
        # conv_input [batch_size, src_len, hid_dim]
        conv_input = self.emb2hid(embedded)

        # permute for convolutional layer
        # conv_input [batch_size, hid_dim, src_len]
        conv_input = conv_input.permute(0, 2, 1)

        # convolutional blocks
        for i, conv in enumerate(self.convs):
            # conv: [batch_size, hid_dim, src_len] -> [batch_size, 2 * hid_dim, src_len] # noqa
            # conved [batch_size, 2 * hid_dim, src_len]
            conved = conv(self.dropout(conv_input))

            # pass through GLU activation function
            # conved [batch_size, hid_dim, src_len]
            conved = torch.nn.functional.glu(conved, dim=1)

            # apply residual connection
            # conved [batch_size, hid_dim, src_len]
            conved = (conved + conv_input) * self.scale.to(conved.device)

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # permute and convert back to emb_dim
        # conved [batch_size, src_len, emb_dim]
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # elementwise sum output (conved) and input (embedded)
        # to be used for attention
        # combined [batch_size, src_len, emb_dim]
        combined = (conved + embedded) * self.scale
        return conved, combined


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
        # inputs[seq_size, batch_size] ->
        # -> [seq_size, batch_size, hidden_size], [1, batch_size, hidden_size]
        outputs, hidden = self._rnn(self._emb(inputs), hidden)
        # [seq_size, batch_size, vocab_size], [1, batch_size, hidden_size])
        return self._out(outputs), hidden


class ConvDecoder(torch.nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 device,
                 max_length=100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = torch.nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, emb_dim)

        self.emb2hid = torch.nn.Linear(emb_dim, hid_dim)
        self.hid2emb = torch.nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = torch.nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = torch.nn.Linear(emb_dim, hid_dim)

        self.fc_out = torch.nn.Linear(emb_dim, output_dim)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=hid_dim,
                            out_channels=2 * hid_dim,
                            kernel_size=kernel_size
                            )
            for _ in range(n_layers)])

        self.dropout = torch.nn.Dropout(dropout)

    def attn(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded [batch_size, trg_len, emb_dim]
        # conved [batch_size, hid_dim, trg_len]
        # encoder_conved,encoder_combined [batch_size, src_len, emb_dim]

        # permute and convert back to emb_dim
        # conved_emb [batch_size, trg_len, emb_dim]
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # combined [batch_size, trg_len, emb_dim]
        combined = (conved_emb + embedded) * self.scale

        # energy [batch_size, trg_len, src_len]
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # attention [batch_size, trg_len, src_len]
        attention = torch.nn.functional.softmax(energy, dim=2)

        # attended_encoding [batch_size, trg_len, emd dim]
        attended_encoding = torch.matmul(attention, encoder_combined)

        # convert from emb_dim -> hid_dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding [batch_size, trg_len, hid_dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1))
        attended_combined *= self.scale.to(conved.device)

        # attended_combined [batch_size, hid_dim, trg_len]
        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg [batch_size, trg_len]
        # encoder_conved, encoder_combined [batch_size, src_len, emb_dim]
        batch_size, trg_len = trg.shape

        # create position tensor
        # pos [batch_size, trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        # embed tokens and positions
        # tok_embedded [batch_size, trg_len, emb_dim]
        tok_embedded = self.tok_embedding(trg)
        # pos_embedded [batch_size, trg_len, emb_dim]
        pos_embedded = self.pos_embedding(pos)

        # combine embeddings by elementwise summing
        # embedded [batch_size, trg_len, emb_dim]
        embedded = self.dropout(tok_embedded + pos_embedded)

        # pass embedded through linear layer to go through emb_dim -> hid_dim
        # conv_input [batch_size, trg_len, hid_dim]
        conv_input = self.emb2hid(embedded)

        # permute for convolutional layer
        # conv_input [batch_size, hid_dim, trg_len]
        conv_input = conv_input.permute(0, 2, 1)

        batch_size, hid_dim = conv_input.shape
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(
                batch_size,
                hid_dim,
                self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            # padded_conv_inp [batch_size, hid_dim, trg_len + kernel size - 1]
            padded_conv_inp = torch.cat((padding, conv_input), dim=2)

            # pass through convolutional layer
            # conved [batch_size, 2 * hid_dim, trg_len]
            conved = conv(padded_conv_inp)

            # pass through GLU activation function
            # conved [batch_size, hid_dim, trg_len]
            conved = torch.nn.functional.glu(conved, dim=1)

            # calculate attention
            # attention [batch_size, trg_len, src_len]
            attention, conved = self.attn(embedded,
                                          conved,
                                          encoder_conved,
                                          encoder_combined)

            # apply residual connection
            # conved [batch_size, hid_dim, trg_len]
            conved = (conved + conv_input) * self.scale.to(conved.device)

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # conved [batch_size, trg_len, emb_dim]
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # output [batch_size, trg_len, output dim]
        output = self.fc_out(self.dropout(conved))

        return output, attention


# https://arxiv.org/abs/1409.3215v3 -> GRU + scheduled sampling
class ScheduledSamplingDecoder(torch.nn.Module):
    def __init__(self, vocab_size,
                 emb_dim=128,
                 rnn_hidden_dim=256,
                 num_layers=1,
                 sampling_rate=0.5):
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
        step = inputs[0]
        result = []
        for original in inputs:
            output, hidden = self._rnn(self._emb(step).unsqueeze(0), hidden)
            result.append(output)

            step = original
            if self.training and bool(np.random.binomial(n=1, p=0.5)):
                with torch.no_grad():
                    step = self._out(output.detach()).argmax(-1).squeeze(0)

        outputs = torch.cat(result)
        return self._out(outputs), hidden


# https://arxiv.org/abs/1406.1078 -> + scheduled sampling
class ExplicitConditioningDecoder(torch.nn.Module):
    def __init__(self, vocab_size,
                 emb_dim=128,
                 rnn_hidden_dim=256,
                 num_layers=1,
                 sampling_rate=0.5):
        super().__init__()

        # self.p = sampling_rate
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim + rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers
        )
        self._out = torch.nn.Linear(rnn_hidden_dim * 2 + emb_dim, vocab_size)

    def forward(self, inputs, encoder_output, encoder_mask, hidden=None):
        step = inputs[0]
        context = encoder_output[[-1]]
        result = []
        for original in inputs:
            emb = self._emb(step).unsqueeze(0)
            output, hidden = self._rnn(
                torch.cat((emb, context), dim=2), hidden)

            output = torch.cat((
                emb.squeeze(0),
                hidden.squeeze(0),
                context.squeeze(0)),
                dim=1
            )
            result.append(output.unsqueeze(0))

            step = original
            if self.training and bool(np.random.binomial(n=1, p=0.5)):
                with torch.no_grad():
                    step = self._out(output.detach()).argmax(-1).squeeze(0)

        outputs = torch.cat(result)
        return self._out(outputs), hidden


def attentiondecoder(atype):
    return partial(AttentionDecoder, atype=atype)


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

    def forward(self, source, target):
        # [batch_size, seq_size] -> [seq_size, batch_size]
        source, target = source.T, target.T
        encoder_mask = (source == 1.)  # find mask for padding inputs

        # output[seq_size, batch_size, hidden_size]
        # hidden[1, batch_size, hidden_size]
        output, hidden = self.encoder(source)
        return self.decoder(target, output, encoder_mask, hidden)


def shift(seq, by, batch_dim=1):
    return torch.cat((seq[by:], seq.new_ones(by, seq.shape[batch_dim])))


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

                last_pred = output[[-1]]
                target = torch.cat([target, last_pred.argmax(-1)], dim=-1)

            # Ensure the sequence has an end
            sentences = target.numpy()
            sentences[:, -1] = tg.vocab.stoi[tg.eos_token]
            yield data, sentences

    def _beam_decode_iterator(self, X, max_len=100, n_beams=3):
        self.module_.eval()
        dataset = self.get_dataset(X)
        tg = X.fields["target"]
        init_token_idx = tg.vocab.stoi[tg.init_token]

        Beam = namedtuple("Beam", ("seq", "score", "hidden"))

        for (data, _) in self.get_iterator(dataset, training=False):
            source = data["source"].T
            source_mask = (source == 1)
            with torch.no_grad():
                enc_src, hidden = self.module_.encoder(source)

            target = source.new_ones(source.shape[1], 1) * init_token_idx
            beams = [Beam(target, 0, hidden)]
            for i in range(max_len + 1):
                new_beams = []
                for beam in beams:
                    with torch.no_grad():
                        # beam.seq[batch_size, seq_size] -> [seq_size, batch]
                        output, hidden = self.module_.decoder(
                            beam.seq.T[[-1]],  # Take the last token [1, batch]
                            enc_src,
                            source_mask,
                            beam.hidden
                        )
                        # output[seq_size, batch_size, vocab_size]

                    # step[seq_size, batch_size, vocab_size]
                    step = torch.nn.functional.log_softmax(output, -1)
                    # vals[1, 1, n_beams], pos[1, 1, n_beams]
                    vals, pos = torch.topk(step, n_beams, dim=-1)

                    for i in range(n_beams):
                        # [batch_size, seq_size] -> [batch_size, seq_size + 1]
                        seq = torch.cat([beam.seq, pos[:, :, i]], -1)
                        score = beam.score - vals[0, 0, i].item()
                        new_beams.append(Beam(seq, score, hidden))
                beams = sorted(new_beams, key=attrgetter("score"))[:n_beams]
            target = beams[0].seq

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

    def score(self, X, y=None, max_len=10):
        tg = X.fields["target"]
        y_true, pred = [], []
        for X, sentences in self._decode_iterator(X, max_len):
            for seq in sentences[:, 1:]:
                stop = np.argmax(seq == tg.vocab.stoi[tg.eos_token])
                pred.append(seq[: stop].tolist())
                y_true.append(X["target"].tolist())

        return corpus_bleu(y_true, pred) * 100


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
        # net.set_params(module__source_pad_idx=svocab["<pad>"])
        # net.set_params(module__target_pad_idx=tvocab["<pad>"])
        net.set_params(criterion__ignore_index=tvocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if m.data.dim() > 1:
        torch.nn.init.xavier_uniform_(m.data)


def build_model(module=TranslationModel, ptype=TextPreprocessor):
    model = LanguageModelNet(
        module=module,
        module__source_vocab_size=1,  # Dummy size
        module__target_vocab_size=1,  # Dummy size
        # module__source_pad_idx=0,  # Dummy size
        optimizer=torch.optim.Adam,
        optimizer__lr=0.01,
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
            DynamicVariablesSetter(),
            skorch.callbacks.GradientNormClipping(1.),
            skorch.callbacks.Initializer("*", fn=initialize_weights),
        ],
    )

    full = make_pipeline(
        build_preprocessor(ptype),
        model,
    )
    return full


def build_preprocessor(ptype=TextPreprocessor,
                       init_token="<s>",
                       eos_token="</s>"):
    source = Field(
        batch_first=True,
        tokenize="spacy",
        init_token=None,
        eos_token=eos_token,
    )

    target = Field(
        batch_first=True,
        tokenize="moses",
        init_token=init_token,
        eos_token=eos_token
    )

    fields = [
        ("source", source),
        ("target", target),
    ]
    return ptype(fields)


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

    bpe_model = build_model(ptype=SubwordPreprocessor)
    validate("BPE encoding", bpe_model, train, test, sample)

    stype = partial(TranslationModel, decodertype=AttentionDecoder)
    aamodel = build_model(mtype=stype)
    validate("Additive attention", aamodel, train, test, sample)

    stype = partial(TranslationModel,
                    decodertype=attentiondecoder(DotAttention))
    aamodel = build_model(mtype=stype)
    validate("Dot attention", aamodel, train, test, sample)

    stype = partial(TranslationModel,
                    decodertype=attentiondecoder(MultiplicativeAttention))
    aamodel = build_model(mtype=stype)
    validate("Multiplicative attention", aamodel, train, test, sample)


if __name__ == "__main__":
    main()
