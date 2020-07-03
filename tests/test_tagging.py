import pytest

from models.tagging import to_pandas
from models.tagging import build_preprocessor, build_preprocessor_emb
from models.tagging import build_model, build_emb_model, build_bert_model
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


@pytest.mark.parametrize("build", [
    build_preprocessor,
    # build_preprocessor_emb, # CI performance issues
])
def test_preprocessing(build, data, batch_size=64):
    dset = build().fit_transform(data)
    batch = next(iter(BucketIterator(dset, batch_size=batch_size)))

    assert batch.tokens.shape[1] == batch_size
    assert batch.tags.shape[1] == batch_size


@pytest.mark.parametrize("build", [
    build_model,
    # build_emb_model # CI performance issues,
    build_bert_model,
])
def test_tagger_model(build, data):
    model = build().fit(data)
    print(model.score(data, data["tags"].values))
