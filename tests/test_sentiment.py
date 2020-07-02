import pytest
import pandas as pd

from torchtext.data import BucketIterator
from models.sentiment import build_preprocessor, build_model
from models.sentiment import VanilaRNN, LSTM


@pytest.fixture
def data(size=320):
    df = pd.DataFrame({
        "review": ["This is good", "This is bad"] * size,
        "sentiment": ["positive", "negative"] * size,
    })
    return df


def test_preprocessing(data, batch_size=32):
    print(data)
    dataset = build_preprocessor().fit_transform(data)

    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.review.shape[-1] == batch_size
    assert batch.sentiment.shape[-1] == batch_size


@pytest.mark.parametrize("module", [
    VanilaRNN,
    LSTM
])
def test_model(module, data):
    build_model(module=module).fit(data)
