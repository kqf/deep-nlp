import torch
import pytest

from models.charrnn import SimpleRNNModel, MemorizerModel
from models.charrnn import build_model


@pytest.fixture
def n_classes():
    return 10


@pytest.fixture
def batch(n_classes, seq_len=5, batch_size=128):
    size = (seq_len, batch_size, 1)
    return torch.randint(0, n_classes, size=size, dtype=torch.long)


def test_simple_model(batch, hidden_size=3):
    model = SimpleRNNModel(batch.shape[-1], hidden_size)
    logits = model(batch.float())
    assert logits.shape == (batch.shape[1], hidden_size)


def test_memorizer_model(batch, n_classes, hidden_size=3):
    model = MemorizerModel(embedding_size=n_classes, hidden_size=hidden_size)
    logits = model(batch)
    assert logits.shape == (batch.shape[1], n_classes)


def test_model():
    build_model().fit([10] * 100, None)
