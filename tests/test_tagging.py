import torch
import pytest

from models.tagging import Tokenizer, iterate_batches
from models.tagging import LSTMTagger, build_model


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


def test_lstm_tagger(raw_data, batch_size=4):
    tt = Tokenizer().fit(raw_data)
    batches = iterate_batches(tt.transform(raw_data), batch_size=batch_size)
    model = LSTMTagger(len(tt.word2ind), len(tt.tag2ind))
    for X, _ in batches:
        logits = model(torch.LongTensor(X))
        seq_len, batch_size = X.shape
        assert logits.shape == (seq_len, batch_size, len(tt.tag2ind))


def test_tagger_model(raw_data):
    model = build_model()
    model.fit(raw_data)
