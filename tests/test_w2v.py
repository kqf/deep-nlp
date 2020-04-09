import pytest

from models.w2v import quora_data, Tokenizer


@pytest.fixture
def data(size=5000):
    return quora_data()[:size]


def test_tokenizer(data):
    tokenizer = Tokenizer().fit(data)
    assert tokenizer is not None
