import torch
import pytest
import pandas as pd

from models.multichoice import Tokenizer, build_vectorizer
from models.multichoice import DSSMEncoder
from models.multichoice import similarity, triplet_loss
from models.multichoice import build_model


@pytest.fixture
def data(size=100):
    df = pd.DataFrame({
        "question": ["Who am I?", "What I like?"] * size,
        "options": [["Rob.", "Bob.", "Ron."], ["cat.", "rum.", "car."]] * size,
        "correct_indices": [[0], [0]] * size,
        "wrong_indices": [[1, 2], [1, 2]] * size,
    })
    return df


def test_tokenizes_sample_data(data):
    tt = Tokenizer("question", "options").fit_transform(data)
    assert tt.shape == data.shape


def test_similarity(batch_size=512, emb_dim=32):
    query = torch.randint(0, 10, (batch_size, emb_dim))
    target = torch.randint(0, 10, (batch_size, emb_dim))

    qt_similarity = similarity(query, target)
    assert qt_similarity.shape == (batch_size,)


def test_triplet_loss(batch_size=512, emb_dim=32):
    query = torch.randint(0, 10, (batch_size, emb_dim))
    correct = torch.randint(0, 10, (batch_size, emb_dim))
    wrong = torch.randint(0, 10, (batch_size, emb_dim))

    qt_similarity = triplet_loss(query, correct, wrong)
    assert qt_similarity.shape == (batch_size,)


def test_dssm_encoder(batch_size=512, seq_size=100,
                      vocab_size=1000, emb_dim=32, output_dim=128):
    queries = torch.randint(0, vocab_size, (batch_size, seq_size))
    embbedings = torch.randn((vocab_size, emb_dim))

    encode = DSSMEncoder(embbedings, output_dim=output_dim)
    assert encode(queries).shape == (batch_size, output_dim)


def test_vectorizes_sample_data(data, batch_size=64):
    tt = build_vectorizer(min_freq=1).fit_transform(data)
    titer = tt(batch_size, torch.device("cpu"))
    batch = next(iter(titer))

    assert batch["questions"].shape[0] == batch_size
    assert batch["correct_answers"].shape[0] == batch_size
    assert batch["wrong_answers"].shape[0] == batch_size


def test_multichoice_model(data):
    build_model(min_freq=1, epochs_count=2).fit(data)
