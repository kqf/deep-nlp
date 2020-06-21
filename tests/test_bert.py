import pytest
import pandas as pd


@pytest.fixture
def data(size=200):
    dataset = {
        "question1": ["How are you?", "Where are you?"] * size,
        "question2": ["How do you do?", "Where is Australia?"] * size,
        "is_duplicate": [1, 0] * size,
    }
    return pd.DataFrame(dataset)


def test_preprocessing(data):
    print(data)
