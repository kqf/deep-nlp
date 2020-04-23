import torch
import pytest
import numpy as np
from sklearn.metrics import f1_score
from models.charrnn import generate_data, BasicRNNClassifier
from models.charrnn import SimpleRNNModel, MemorizerModel

torch.manual_seed(0)
np.random.seed(0)


@pytest.fixture
def batch(seq_len=5, batch_size=128):
    X = torch.randint(0, 10, size=(seq_len, batch_size, 1), dtype=torch.long)
    return X


def test_simple_model(batch, hidden_size=3):
    model = SimpleRNNModel(batch.shape[-1], hidden_size)

    logits = model(batch.float())
    logits.shape == (batch.shape[0], hidden_size)


def test_memorizer_model(batch, hidden_size=3):
    model = MemorizerModel(embedding_size=10, hidden_size=hidden_size)
    logits = model(batch)
    # It should have hidden_size = output_size
    logits.shape == (batch.shape[0], hidden_size)


@pytest.fixture
def sample(seq_len=25):
    return list(zip(*generate_data(seq_len=25)))


def test_char_rnn(sample):
    X, y = sample

    model = BasicRNNClassifier()
    model.fit(X, y)
    assert model.predict_proba(X).shape == (len(X), len(set(y)))

    y_pred = model.predict(X)
    assert y_pred.shape == (len(X),)

    # In most of the cases this should work
    assert f1_score(y, y_pred, average="micro") > 0.95
