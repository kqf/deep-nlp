import pytest
from models.charrnn import generate_data, BasicRNNClassifier


@pytest.fixture
def fake_data():
    return list(zip(*generate_data()))


def test_dummy_rnn(fake_data):
    X, y = fake_data

    model = BasicRNNClassifier()
    model.fit(X, y)
