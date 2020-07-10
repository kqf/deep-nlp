import pytest

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from models.lstm import build_model
from models.lstm import build_preprocessor

from torchtext.data import BucketIterator

torch.manual_seed(0)
np.random.seed(0)


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


def test_surname_classifier(data):
    model = build_model()
    model.fit(data)

    assert f1_score(data["labels"], model.predict(data)) > 0.95
