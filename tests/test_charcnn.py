import pytest
import numpy as np
import pandas as pd
from models.charcnn import custom_f1, build_model
from models.charcnn import build_preprocessor

from sklearn.metrics import f1_score
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=128):
    return pd.DataFrame({
        "surname": ["smith", "aardwark", "chair"] * size,
        "label": [1, 0, 0] * size
    })


def test_generates_batches(data, batch_size=128):
    dset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dset, batch_size=batch_size)))

    assert batch.surname.shape[0] == batch_size


def test_model(data):
    model = build_model().fit(data)
    assert model is not None

    probs = model.predict_proba(data)
    assert probs.shape == (data.shape[0], 2)

    y_pred = model.predict(data)
    np.testing.assert_array_equal(y_pred, data["label"].values)


@pytest.mark.parametrize("y_pred, y", [
    ([1, 1, 1, 1], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [1, 1, 1, 1]),
    ([1, 0, 1, 0], [1, 0, 1, 1]),
])
def test_custom_f1(y_pred, y):
    y_pred, y = np.array(y_pred), np.array(y)
    assert custom_f1(y_pred, y) == f1_score(y_pred, y)
