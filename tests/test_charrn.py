import torch
import pytest
from models.charrnn import generate_data, BasicRNNClassifier
from models.charrnn import SimpleRNN


@pytest.fixture
def fake_data(seq_len=25):
    return list(zip(*generate_data(seq_len=25)))


def test_dummy_rnn(fake_data):
    X, y = fake_data

    model = BasicRNNClassifier()
    model.fit(X, y)


def test_simple_model(batch_size=128, seq_len=5, hidden_size=3):
    X = torch.randint(0, 10, size=(seq_len, batch_size, 1), dtype=torch.long)
    model = SimpleRNN(1, hidden_size)

    logits = model(X.float())
    logits.shape == (batch_size, hidden_size)
