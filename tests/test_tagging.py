import pytest
import gensim.downloader as gapi

from models.tagging import build_preprocessor, to_pandas
from models.tagging import LSTMTagger, build_model, BiLSTMTagger
from models.tagging import EmbeddingsTokenizer
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=64):
    nltk_raw = [[
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

    return to_pandas(nltk_raw)


def test_preprocessing(data, batch_size=64):
    dset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dset, batch_size=batch_size)))

    assert batch.tokens.shape[1] == batch_size
    assert batch.tags.shape[1] == batch_size


@pytest.mark.skip
@pytest.fixture
def w2v():
    return gapi.load('glove-wiki-gigaword-100')


@pytest.mark.skip
def test_embedding_tokenizer(w2v, data):
    tokenizer = EmbeddingsTokenizer(w2v).fit(data)
    tokenizer.emb_size.shape[1] == 100


@pytest.mark.parametrize("model_type", [
    LSTMTagger,
    BiLSTMTagger,
])
def test_lstm_tagger(model_type, data, batch_size=4):
    dset = build_preprocessor().fit_transform(data)
    vocab_size = len(dset.fields["tokens"].vocab)
    tags_size = len(dset.fields["tags"].vocab)

    batch = next(iter(BucketIterator(dset, batch_size=batch_size)))
    model = model_type(vocab_size, tags_size)
    logits = model(batch.tokens)

    seq_len, batch_size = batch.tokens.shape
    assert logits.shape == (seq_len, batch_size, tags_size)


def test_tagger_model(data):
    model = build_model()
    model.fit(data)
