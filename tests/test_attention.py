import torch
import pytest
import pandas as pd

from torchtext.data import BucketIterator

from models.attention import build_preprocessor, build_model
from models.attention import PositionalEncoding, LayerNorm


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


def test_full_model(data):
    model = build_model().fit(data)
    assert model is not None
