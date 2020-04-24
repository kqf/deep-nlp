import pytest
import pandas as pd
from models.lstm import Tokenizer


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
