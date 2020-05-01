import pytest
import torch
import pandas as pd

from models.nlg import LockedDropout
from models.nlg import ConvLM, RnnLM, generate, TextTransformer, build_model


@pytest.fixture
def batch(vocab_size, seq_length, batch_size):
    return torch.randint(0, vocab_size, (seq_length, batch_size))


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("vocab_size", [26])
def test_cnn_lm(batch, vocab_size, seq_length, batch_size):
    model = ConvLM(vocab_size)
    assert model(batch)[0].shape == (seq_length, batch_size, vocab_size)


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
def test_rnn_lm(batch, vocab_size, seq_length, batch_size):
    model = RnnLM(vocab_size)
    assert model(batch)[0].shape == (seq_length, batch_size, vocab_size)

    assert len(list(generate(model))) == 150


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
    return pd.Series([sample] * 1000)


def test_text_transformer(corpus):
    tt = TextTransformer().fit(corpus)
    assert len(tt.transform(corpus)) == len(corpus)


def test_nlg_training_loop(corpus):
    model = build_model()
    model.fit(corpus, None)

    generated = model.inverse_transform([[0.7, 100]])
    assert len(generated) == 1, "generating just a phrase"
