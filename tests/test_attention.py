import pytest
import pandas as pd

from torchtext.data import BucketIterator

from models.attention import build_preprocessor, build_model


@pytest.fixture
def data(size=128):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target":
        ["Tout le travail et aucun jeu font de Jack un garçon terne"] * size,
    }
    return pd.DataFrame(corpus)


def test_textpreprocessor(data, batch_size=128):
    tp = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(tp, batch_size)))
    # assert tp.transform(data) is not None
    assert batch.source.shape[0] == batch_size
    assert batch.target.shape[0] == batch_size


def test_full_model(data):
    model = build_model().fit(data)
    assert model is not None
