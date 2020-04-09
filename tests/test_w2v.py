import pytest

from models.w2v import quora_data, Tokenizer, build_contexts


@pytest.fixture
def data(size=5000):
    return quora_data()[:size]


def test_tokenizer(data):
    tokenizer = Tokenizer().fit(data)
    assert tokenizer is not None


def test_contexts():
    inputs = [[1, 2, 3, 4, 5]]
    outputs = [
        (1, [2, 3]),
        (2, [1, 3, 4]),
        (3, [1, 2, 4, 5]),
        (4, [2, 3, 5]),
        (5, [3, 4])
    ]
    assert list(build_contexts(inputs, window_size=2)) == outputs
