import pytest
from models.elmo import build_preprocessor
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=100):
    example = (
        [
            'All', 'work', 'and', 'no', 'play', 'makes',
            'Jack', 'a', 'dull', 'boy'
        ],
        ['O', 'O', 'O', 'O', 'O', 'O', 'I-PER', 'O', 'O', 'O'],
    )
    return [example, ] * size


def test_prepocessor(data, batch_size=10):
    tp = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(tp, batch_size)))

    assert batch.tokens.shape[0] == batch_size
    assert batch.tags.shape[0] == batch_size
