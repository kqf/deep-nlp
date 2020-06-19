import pytest

from torchtext.data import BucketIterator
from models.embeddings import build_preprocessor


@pytest.fixture
def data(size=100):
    return [
        "All work and no play makes Jack a dull boy",
        "I am sorry Dave I am afraid I can not do that",
    ] * size


def test_representation(data, batch_size=64):
    dataset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))

    assert batch.context.shape == (batch_size, 1)
    assert batch.target.shape == (batch_size, 1)
