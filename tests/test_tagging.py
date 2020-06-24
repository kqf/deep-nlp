import torch
import pytest
import gensim.downloader as gapi
from models.tagging import iterate_batches
from models.tagging import LSTMTagger, build_model, BiLSTMTagger
from models.tagging import EmbeddingsTokenizer, Tokenizer


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


@pytest.fixture
def w2v():
    return gapi.load('glove-wiki-gigaword-100')


def test_embedding_tokenizer(w2v, raw_data):
    tokenizer = EmbeddingsTokenizer(w2v).fit(raw_data)
    tokenizer.emb_size.shape[1] == 100


def test_iterates_the_batches(data, batch_size=4):
    batches = iterate_batches(data, batch_size=batch_size)
    for X, y in batches:
        assert X.shape[1] == batch_size
        assert y.shape[1] == batch_size


@pytest.mark.parametrize("model_type", [
    LSTMTagger,
    BiLSTMTagger,
])
def test_lstm_tagger(model_type, raw_data, batch_size=4):
    tt = Tokenizer().fit(raw_data)
    batches = iterate_batches(tt.transform(raw_data), batch_size=batch_size)
    model = model_type(len(tt.word2ind), len(tt.tag2ind))
    for X, _ in batches:
        logits = model(torch.LongTensor(X))
        seq_len, batch_size = X.shape
        assert logits.shape == (seq_len, batch_size, len(tt.tag2ind))


def test_tagger_model(raw_data):
    model = build_model(epochs_count=2)
    model.fit(raw_data)
