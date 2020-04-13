import pytest

from models.w2v import quora_data, Tokenizer
from models.w2v import SkipGram, CBoW, Word2VecGeneric


@pytest.fixture
def data(size=5000):
    return quora_data()[:size]


def test_tokenizer(data):
    tokenizer = Tokenizer().fit(data)
    assert tokenizer.transform(data) is not None


def test_contexts():
    inputs = [[1, 2, 3, 4, 5]]
    outputs = [
        (1, [2, 3]),
        (2, [1, 3, 4]),
        (3, [1, 2, 4, 5]),
        (4, [2, 3, 5]),
        (5, [3, 4])
    ]
    contexts = list(Word2VecGeneric.build_contexts(inputs, window_size=2))
    assert list(contexts) == outputs

    batches = list(SkipGram.batches(
        contexts,
        window_size=2,
        num_skips=4,
        batch_size=4))

    assert tuple(map(set, batches[0])) == ({3}, {1, 2, 4, 5})
    assert len(batches) == 1

    batches = list(CBoW.batches(contexts, window_size=2, batch_size=4))

    assert len(batches) == 1
    inputs, label = batches[0]
    assert inputs.tolist() == [[1, 2, 4, 5]]
    assert label.tolist() == [3]


def test_skipgram():
    inputs = [[1, 2, 3, 4, 5]] * 1000
    model = SkipGram().fit(inputs)

    assert model.embeddings_ is not None


def test_cbow():
    inputs = [[1, 2, 3, 4, 5]] * 1000
    model = CBoW().fit(inputs)

    assert model.embeddings_ is not None
