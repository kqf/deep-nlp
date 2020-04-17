import pytest
import numpy as np
import pandas as pd
from models.charcnn import Tokenizer, CharClassifier, custom_f1
from sklearn.metrics import f1_score


@pytest.fixture
def data(size=1000):
    return pd.DataFrame({
        "names": ["smith", "aardwark", "chair"] * size,
        "labels": [1, 0, 0] * size
    })


def test_converts_data(data):
    tokenizer = Tokenizer().fit(data["names"])
    assert tokenizer.max_len == 8

    tokenized = tokenizer.transform(data["names"])
    assert tokenized.shape, (data.shape, tokenizer.max_len)


@pytest.mark.parametrize("bsize", [1, 2, 3, 4, 31, 128])
def test_generates_batches(data, bsize):
    x, y = next(CharClassifier.batches(data["names"], data["labels"], bsize))
    assert len(x) == len(y)


def test_charclassifier(data):
    model = CharClassifier().fit(data["names"], data["labels"].values)
    assert model is not None

    probs = model.predict_proba(data["names"])
    assert probs.shape == (data.shape[0], 2)

    y_pred = model.predict(data["names"])
    np.testing.assert_array_equal(y_pred, data["labels"].values)


@pytest.mark.parametrize("y_pred, y", [
    ([1, 1, 1, 1], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [1, 1, 1, 1]),
    ([1, 0, 1, 0], [1, 0, 1, 1]),
])
def test_custom_f1(y_pred, y):
    y_pred, y = np.array(y_pred), np.array(y)
    assert custom_f1(y_pred, y) == f1_score(y_pred, y)
