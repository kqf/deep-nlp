import pytest
import pandas as pd
from models.charcnn import Tokenizer, CharClassifier


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
