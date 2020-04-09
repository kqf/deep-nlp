import pytest

from models.w2v import quora_data, Tokenizer, build_contexts
from models.w2v import skip_gram_batchs


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
    contexts = list(build_contexts(inputs, window_size=2))
    assert list(contexts) == outputs

    batches = skip_gram_batchs(
        contexts,
        window_size=2,
        num_skips=4,
        batch_size=4)
    assert tuple(map(set, next(batches))) == ({3}, {1, 2, 4, 5})
