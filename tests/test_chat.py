import torch
import pytest
import pandas as pd

from models.chat import Tokenizer, build_vectorizer
from models.chat import similarity, triplet_loss


@pytest.fixture
def data(size=100):
    df = pd.DataFrame({
        "question": ["Who am I?", "What I like?"] * size,
        "options": [["Rob.", "Bob.", "Ron."], ["cat.", "rum.", "car."]] * size,
        "correct_indices": [[1], [1]] * size,
        "wrong_indices": [[2, 3], [2, 3]] * size,
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


def test_vectorizes_sample_data(data):
    tt = build_vectorizer().fit_transform(data)
    assert tt is not None
