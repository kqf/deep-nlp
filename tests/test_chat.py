import pytest
import pandas as pd

from models.chat import Tokenizer, build_vectorizer


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


def test_vectorizes_sample_data(data):
    tt = build_vectorizer().fit_transform(data)
    assert tt is not None
