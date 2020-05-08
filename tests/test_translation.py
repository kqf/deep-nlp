import pytest
import torch
import pandas as pd

from models.translation import TextPreprocessor
from models.translation import Encoder, Decoder, TranslationModel, build_model


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
def source(source_vocab_size, source_seq_size, batch_size):
    return torch.randint(0, source_vocab_size, (source_seq_size, batch_size))


@pytest.fixture
def target(target_vocab_size, target_seq_size, batch_size):
    return torch.randint(0, target_vocab_size, (target_seq_size, batch_size))


@pytest.mark.parametrize("source_seq_size", [120])
@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("source_vocab_size", [26])
@pytest.mark.parametrize("rnn_hidden_dim", [256])
def test_encoder(source, source_vocab_size, batch_size, rnn_hidden_dim):
    enc = Encoder(source_vocab_size, rnn_hidden_dim)
    # Return the last hidden state source_seq_size -> 1
    assert enc(source).shape == (1, batch_size, rnn_hidden_dim)


@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("source_seq_size", [121])
@pytest.mark.parametrize("target_seq_size", [122])
@pytest.mark.parametrize("source_vocab_size", [26])
@pytest.mark.parametrize("target_vocab_size", [33])
def test_decoder(
        source, target,
        source_vocab_size, target_vocab_size,
        batch_size, target_seq_size):
    encode = Encoder(source_vocab_size)
    decode = Decoder(target_vocab_size)
    output, hidden = decode(target, encode(source))
    assert output.shape == (target_seq_size, batch_size, target_vocab_size)


@pytest.mark.parametrize("batch_size", [128, 512])
@pytest.mark.parametrize("source_seq_size", [121])
@pytest.mark.parametrize("target_seq_size", [122])
@pytest.mark.parametrize("source_vocab_size", [26])
@pytest.mark.parametrize("target_vocab_size", [33])
def test_translation_model(
        source, target,
        source_vocab_size, target_vocab_size,
        batch_size, target_seq_size):
    translate = TranslationModel(source_vocab_size, target_vocab_size)
    output, hidden = translate(source, target)
    assert output.shape == (target_seq_size, batch_size, target_vocab_size)


@pytest.fixture
def examples():
    data = {
        "source": ["All work"],
        "target": ["Tout"],
    }
    return pd.DataFrame(data)


def test_translates(data, examples):
    model = build_model()
    model.fit(data, None)
    print(model.transform(examples))
