import pytest

from models.tagging import Tokenizer, iterate_batches


@pytest.fixture
def raw_data(size=100):
    return [[
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
    ]] * size


@pytest.fixture
def data(raw_data):
    tokenizer = Tokenizer().fit(raw_data)
    return tokenizer.transform(raw_data)


def test_iterates_the_batches(data, batch_size=4):
    batches = iterate_batches(data, batch_size=batch_size)
    for X, y in batches:
        assert X.shape[1] == batch_size
        assert y.shape[1] == batch_size
