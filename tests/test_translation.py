import pytest
import torch
import pandas as pd

from models.translation import TextPreprocessor, SubwordTransformer
from models.translation import Encoder, Decoder, AttentionDecoder
from models.translation import TranslationModel
from models.translation import build_model, build_model_bpe
from models.translation import ScheduledSamplingDecoder


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


def test_subwordtransformer(data):
    tp = SubwordTransformer(num_symbols=4).fit(data)
    assert tp.transform(data) is not None


@pytest.fixture
def source(source_vocab_size, source_seq_size, batch_size):
    return torch.randint(0, source_vocab_size, (source_seq_size, batch_size))


@pytest.fixture
def target(target_vocab_size, target_seq_size, batch_size):
    return torch.randint(0, target_vocab_size, (target_seq_size, batch_size))


@pytest.mark.parametrize("source_seq_size", [120])
@pytest.mark.parametrize("batch_size", [512])
@pytest.mark.parametrize("source_vocab_size", [26])
@pytest.mark.parametrize("rnn_hidden_dim", [256])
def test_encoder(source, source_seq_size,
                 source_vocab_size, batch_size, rnn_hidden_dim):
    enc = Encoder(source_vocab_size, rnn_hidden_dim)
    # Return the last hidden state source_seq_size -> 1
    encoded, hidden = enc(source)
    assert hidden.shape == (1, batch_size, rnn_hidden_dim)
    assert encoded.shape == (source_seq_size, batch_size, rnn_hidden_dim)


@pytest.mark.parametrize("batch_size", [32, 512])
@pytest.mark.parametrize("source_seq_size", [121])
@pytest.mark.parametrize("target_seq_size", [122])
@pytest.mark.parametrize("source_vocab_size", [26])
@pytest.mark.parametrize("target_vocab_size", [33])
@pytest.mark.parametrize("decodertype", [
    Decoder,
    ScheduledSamplingDecoder,
    AttentionDecoder,
])
def test_decoder(
        source, target,
        source_vocab_size, target_vocab_size,
        batch_size, target_seq_size, decodertype):
    encode = Encoder(source_vocab_size)
    decode = decodertype(target_vocab_size)

    encoded, hidden = encode(source)
    mask = (source == 1)

    output, hidden, weights = decode(target, encoded, mask, hidden)
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
    output, hidden, weights = translate(source, target)
    assert output.shape == (target_seq_size, batch_size, target_vocab_size)


@pytest.fixture
def examples():
    data = {
        "source": ["All work"],
        "target": [""],
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("create_model", [
    build_model,
    build_model_bpe,
])
def test_translates(create_model, data, examples):
    model = create_model()
    # First fit the text pipeline
    text = model[0]
    text.fit(data, None)
    # Then use to initialize the model
    model[-1].model_init(
        source_vocab_size=len(text[-1].source.vocab),
        target_vocab_size=len(text[-1].target.vocab),
    )
    # Now we are able to generate from the untrained model
    print("Before training")
    print(model.transform(examples))

    model.fit(data, None)
    print("After training")
    print(model.transform(examples))
    model.n_beams = 5
    print("After training, beam search")
    print(model.transform(examples))
    assert model.score(examples) > -1
