import pytest

from torchtext.data import BucketIterator
from models.embeddings import build_preprocessor, build_model
from models.embeddings import NegativeSamplingIterator


@pytest.fixture
def data(size=1000):
    return [
        "All work and no play makes Jack a dull boy",
        # "I am sorry Dave I am afraid I can not do that",
    ] * size


def test_representation(data, batch_size=64):
    dataset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))

    assert batch.context.shape == (batch_size, 1)
    assert batch.target.shape == (batch_size, 1)


def test_negative_sampling_iterator(data, batch_size=64, neg_samples=5):
    dataset = build_preprocessor().fit_transform(data)

    ns = NegativeSamplingIterator(
        neg_samples, 3. / 4., dataset, batch_size=batch_size)

    batch, _ = next(iter(ns))

    assert batch["context"].shape == (batch_size,)
    assert batch["target"].shape == (batch_size,)
    assert batch["negatives"].shape == (batch_size, neg_samples)


def test_model(data):
    model = build_model()
    model.fit(data)
