import pytest

from models.tagging import Tokenizer


@pytest.fixture
def raw_data(size=100):
    return [
        ('The', 'DET'),
        ('grand', 'ADJ'),
        ('jury', 'NOUN'),
        ('commented', 'VERB'),
        ('on', 'ADP'),
        ('a', 'DET'),
        ('number', 'NOUN'),
        ('of', 'ADP'),
        ('other', 'ADJ'),
        ('topics', 'NOUN'),
        ('.', '.')
    ]


@pytest.fixture
def data(raw_data):
    tokenizer = Tokenizer().fit(raw_data)
    return tokenizer.transform(raw_data)


def test_data(data):
    print(data)
