import pytest
import torch

from torchtext.data import BucketIterator

from models.elmo import build_preprocessor
from models.elmo import build_baseline, build_elmo, build_crf
from models.elmo import BaselineTagger


@pytest.fixture
def data(size=160):
    example1 = (
        [
            'All', 'work', 'and', 'no', 'play', 'makes',
            'Jack', 'a', 'dull', 'boy'
        ],
        ['O', 'O', 'O', 'O', 'O', 'O', 'I-PER', 'O', 'O', 'O'],
    )
    example2 = (
        ["The", "chocolate", "makes", "Andrew", "happy"],
        ["O", "O", "O", "I-PER", "O"],
    )
    return [example1, example2] * size


@pytest.fixture
def batch(data, batch_size=160):
    tp = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(tp, batch_size)))

    assert batch.tokens.shape[0] == batch_size
    assert batch.tags.shape[0] == batch_size

    return batch


@pytest.fixture
def embeddings(vocab_size=100, emb_dim=100):
    return torch.rand(vocab_size, emb_dim)


def test_baseline_module(embeddings, batch, n_tags=2):
    model = BaselineTagger(embeddings, n_tags)
    logits = model(batch.tokens)

    batch_size, seq_len = batch.tokens.shape
    assert logits.shape == (batch_size, seq_len, n_tags)


@pytest.mark.parametrize("build", [
    build_baseline,
    build_elmo,
    build_crf,
])
def test_baseline_model(build, data):
    model = build().fit(data)

    tokens, tags = zip(*data)
    score = model.score(data, tags)
    print("F1 score: ", score)
    assert score > 0.5
