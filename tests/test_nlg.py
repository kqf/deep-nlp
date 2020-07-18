import pytest
import torch
import pandas as pd

from models.nlg import LockedDropout
from models.nlg import build_preprocessor
from models.nlg import ConvLM, RnnLM, generate, build_model
from torchtext.data import BucketIterator


@pytest.fixture
def batch(vocab_size, seq_length, batch_size):
    return torch.randint(0, vocab_size, (seq_length, batch_size))


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("vocab_size", [26])
def test_cnn_lm(batch, vocab_size, seq_length, batch_size):
    model = ConvLM(vocab_size)
    assert model(batch)[0].shape == (seq_length, batch_size, vocab_size)

    assert len(list(generate(model))) < 150


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
def test_rnn_lm(batch, vocab_size, seq_length, batch_size):
    model = RnnLM(vocab_size)
    assert model(batch)[0].shape == (seq_length, batch_size, vocab_size)

    assert len(list(generate(model))) < 150


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
def test_locked_dropout(batch, vocab_size, seq_length, batch_size):
    embedded = torch.nn.Embedding(vocab_size, 12)(batch)
    model = LockedDropout()
    assert model(embedded).shape == embedded.shape


@pytest.fixture
def corpus():
    sample = (
        "All work and no play makes Jack a dull boy,"
        "All play and no work makes Jack a mere toy."
    )
    return pd.Series([sample] * 320)


@pytest.fixture
def data(corpus):
    return pd.DataFrame({"text": corpus.values})


def test_preprocessing(data, batch_size=32):
    dataset = build_preprocessor().fit_transform(data)

    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size


def test_nlg_training_loop(data):
    model = build_model()
    model.fit(data, None)

    tstart = model[0].fields[0][-1].vocab["<s>"]
    tend = model[0].fields[0][-1].vocab["</s>"]

    generated = model.inverse_transform([[0.1, 100, tstart, tend]])
    print()
    print(generated)
    assert len(generated) == 1, "generating just a phrase"
