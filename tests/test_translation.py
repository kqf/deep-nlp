import pytest
import torch
import pandas as pd

from torchtext.data import BucketIterator
from models.translation import build_preprocessor, SubwordTransformer
from models.translation import Encoder, Decoder, AttentionDecoder
from models.translation import AdditiveAttention, DotAttention
from models.translation import MultiplicativeAttention
from models.translation import TranslationModel
from models.translation import build_model, build_model_bpe
from models.translation import ScheduledSamplingDecoder

from functools import partial


@pytest.fixture
def data(size=100):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target":
        ["Tout le travail et aucun jeu font de Jack un garçon terne"] * size,
    }
    return pd.DataFrame(corpus)


@pytest.mark.parametrize("prep", [
    build_preprocessor(),
])
def test_preprocessing(prep, data, batch_size=32):
    print(data)
    dataset = prep.fit_transform(data)

    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.source.shape[-1] == batch_size
    assert batch.target.shape[-1] == batch_size


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


@pytest.fixture
def attention_inputs(batch_size, query_size, key_size, seq_len):
    query = torch.randn((batch_size, query_size))
    key = torch.randn((seq_len, batch_size, key_size))
    value = torch.randn((seq_len, batch_size, key_size))
    mask = torch.randn((seq_len, batch_size)) > 0
    return query, key, value, mask


@pytest.fixture
def attention_out_shapes(batch_size, query_size, key_size, seq_len):
    context = (batch_size, key_size)
    weights = (seq_len, batch_size, 1)
    return context, weights


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [122])
@pytest.mark.parametrize("query_size", [32])
@pytest.mark.parametrize("key_size", [32])
@pytest.mark.parametrize("hidden_dim", [256])
@pytest.mark.parametrize("attentionlayer", [
    AdditiveAttention,
    DotAttention,
    MultiplicativeAttention,
])
def test_attention(
        attentionlayer, query_size, key_size, hidden_dim,
        attention_inputs, attention_out_shapes):
    attention = attentionlayer(query_size, key_size, hidden_dim)
    output, weights = attention(*attention_inputs)

    output_shape, weights_shape = attention_out_shapes

    assert output.shape == output_shape
    assert weights.shape == weights_shape


@pytest.mark.parametrize("batch_size", [32])
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

    output, hidden = decode(target, encoded, mask, hidden)
    assert output.shape == (target_seq_size, batch_size, target_vocab_size)


@pytest.fixture
def examples():
    data = {
        "source": ["All work"],
        "target": [""],
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("mtype", [
    TranslationModel,
    partial(TranslationModel, decodertype=ScheduledSamplingDecoder),
    partial(TranslationModel, decodertype=AttentionDecoder),
])
@pytest.mark.parametrize("create_model", [
    build_model,
    build_model_bpe,
])
def test_translates(create_model, mtype, data, examples):
    model = create_model(mtype=mtype, epochs_count=2)
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
