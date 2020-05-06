import pytest
import torch
import pandas as pd

from models.translation import TextPreprocessor, Encoder, Decoder


@pytest.fixture
def data(size=100):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target":
        ["Tout le travail et aucun jeu font de Jack un garçon terne"] * size,
    }
    return pd.DataFrame(corpus)


def test_textpreprocessor(data):
    tp = TextPreprocessor().fit(data)
    assert tp.transform(data) is not None


@pytest.fixture
def batch(vocab_size, seq_length, batch_size):
    return torch.randint(0, vocab_size, (seq_length, batch_size))


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
@pytest.mark.parametrize("rnn_hidden_dim", [256])
def test_encoder(batch, vocab_size, batch_size, rnn_hidden_dim):
    enc = Encoder(vocab_size, rnn_hidden_dim)
    # Return the last hidden state seq_length -> 1
    assert enc(batch).shape == (1, batch_size, rnn_hidden_dim)


@pytest.mark.parametrize("seq_length", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("vocab_size", [26])
@pytest.mark.parametrize("rnn_hidden_dim", [256])
def test_decoder(batch, vocab_size, batch_size, rnn_hidden_dim, seq_length):
    enc = Encoder(vocab_size, rnn_hidden_dim)
    dec = Decoder(vocab_size, rnn_hidden_dim)
    output, hidden = dec(batch, enc(batch))
    assert output.shape == (seq_length, batch_size, vocab_size)
