import pytest
import pandas as pd
from models.chat import build_preprocessor
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=128):
    return pd.DataFrame({
        "query": [
            "How are you?",
            "I am fine as well, and where do you live?",
            "Do you like ice cream?",
        ] * size,
        "target": [
            "I am fine thanks, and you?",
            "Good, I live in Lodon.",
            "No",
        ] * size

    })


def test_dummy(data, batch_size=128):
    print(data)
    dataset = build_preprocessor().fit_transform(data)
    data_iter = BucketIterator(dataset, batch_size=batch_size)

    assert next(iter(data_iter)).query.shape[1] == batch_size
    assert next(iter(data_iter)).target.shape[1] == batch_size
