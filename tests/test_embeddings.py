import pytest

from torchtext.data import BucketIterator
from models.embeddings import build_preprocessor
from models.embeddings import build_model, build_sgns_model, build_cbow_model
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

    ns = NegativeSamplingIterator(dataset, batch_size, neg_samples, 3. / 4.)
    batch, _ = next(iter(ns))

    assert batch["context"].shape == (batch_size,)
    assert batch["target"].shape == (batch_size,)
    assert batch["negatives"].shape == (batch_size, neg_samples)


@pytest.mark.parametrize("build", [
    build_model,
    build_cbow_model,
])
def test_model(build, data):
    build().fit(data)


def test_sgns_model(data):
    model = build_sgns_model()
    model.fit(data)
    model.predict_proba(data)
