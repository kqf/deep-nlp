import torch
import pytest
from models.charrnn import generate_data, BasicRNNClassifier
from models.charrnn import SimpleRNN


@pytest.fixture
def fake_data():
    return list(zip(*generate_data()))


def test_dummy_rnn(fake_data):
    X, y = fake_data

    model = BasicRNNClassifier()
    model.fit(X, y)


def test_simple_model(batch_size=128, seq_len=5):
    data = torch.randint(0, 10, size=(seq_len, batch_size), dtype=torch.long)
    model = SimpleRNN(1, 3)

    logits = model(data)

    # logits = model(logits, logits)
