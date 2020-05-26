import pytest
import pandas as pd

from models.chat import Tokenizer


@pytest.fixture
def data(size=100):
    df = pd.DataFrame({
        "question": ["Who am I?", "What I like?"] * size,
        "answers": [["Rob", "Bob", "Ron"], ["cat", "rum", "car"]] * size,
        "correct_indices": [[1], [1]] * size,
        "wrong_indices": [[2, 3], [2, 3]] * size,
    })
    return df


def test_tokenizes_sample_data(data):
    tt = Tokenizer("question", "answers").fit_transform(data)
    assert tt.shape == data.shape
