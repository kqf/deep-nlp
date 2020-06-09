import torch
import pytest
import pandas as pd

from torchtext.data import BucketIterator

from models.attention import build_preprocessor, build_model
from models.attention import PositionalEncoding, LayerNorm
from models.attention import Encoder, Decoder


@pytest.fixture
def data(size=100):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target":
        ["Tout le travail et aucun jeu font de Jack un garçon terne"] * size,
    }
    return pd.DataFrame(corpus)


def test_textpreprocessor(data, batch_size=32):
    tp = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(tp, batch_size)))
    # assert tp.transform(data) is not None
    assert batch.source.shape[0] == batch_size
    assert batch.target.shape[0] == batch_size


def test_positional_encoding(emb_dim=10):
    pe = PositionalEncoding(emb_dim, dropout=0)

    emb = torch.zeros((1, 100, emb_dim))
    assert pe(emb).shape == emb.shape


def test_layer_normalization(n_features=20, batch_size=128):
    ln = LayerNorm(n_features)
    data = torch.rand(batch_size, n_features)
    assert ln(data).shape == data.shape


def test_encoder(seq_size=100, batch_size=128, vocab_size=30, model_d=10):
    enc = Encoder(vocab_size, model_d, 10, 10, 5, 0.7)
    inputs, mask = torch.randint(0, vocab_size, (seq_size, batch_size)), None
    outputs = enc(inputs.T, mask)
    assert outputs.shape == (batch_size, seq_size, model_d)


def test_decoder(
        batch_size=128, model_d=10,
        src_seq_size=200, trg_seq_size=100, trg_vocab_size=40):
    dec = Decoder(trg_vocab_size, model_d, 10, 10, 5, 0.7)

    inp_trg = torch.randint(0, trg_vocab_size, (batch_size, trg_seq_size))
    enc_src = torch.rand(batch_size, src_seq_size, model_d)
    mask = None

    outputs = dec(inp_trg, enc_src, mask, mask)
    assert outputs.shape == (batch_size, trg_seq_size, trg_vocab_size)


# @pytest.mark.skip
def test_full_model(data):
    model = build_model().fit(data)
    assert model is not None
