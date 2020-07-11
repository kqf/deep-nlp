import torch
import pytest
import pandas as pd

from functools import partial
from models.lstm import build_model, SimpleRNNModel
from models.lstm import build_preprocessor
from torchtext.data import BucketIterator

from sklearn.metrics import f1_score


@pytest.fixture
def data(size=128):
    return pd.DataFrame({
        "names": ["smith", "aardwark", "chair"] * size,
        "labels": [1, 0, 0] * size
    })


def test_generates_batches(data, batch_size=128):
    dset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dset, batch_size=batch_size)))

    assert batch.names.shape[1] == batch_size


@pytest.mark.parametrize("rnn_type", [
    SimpleRNNModel,
    torch.nn.LSTM,
    partial(torch.nn.LSTM, bidirectional=True),
])
def test_surname_classifier(rnn_type, data):
    model = build_model(rnn_type=rnn_type).fit(data)
    assert f1_score(data["labels"], model.predict(data)) > 0.95
