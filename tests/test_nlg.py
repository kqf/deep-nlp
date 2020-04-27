import pytest

import torch

from models.nlg import ConvLM, RnnLM, generate


@pytest.fixture
def batch(vocab_size, seq_length, batch_size):
    return torch.randint(0, vocab_size, (seq_length, batch_size))


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("vocab_size", [26])
def test_cnn_lm(batch, vocab_size, seq_length, batch_size):
    model = ConvLM(vocab_size, seq_length=seq_length)
    assert model(batch)[0].shape == (batch_size, vocab_size)

    assert len(list(generate(model))) == 150


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
def test_rnn_lm(batch, vocab_size, batch_size):
    model = RnnLM(vocab_size)
    assert model(batch)[0].shape == (batch_size, vocab_size)
