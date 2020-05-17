import pytest
import pandas as pd

from models.dialogue import build_preprocessor


@pytest.fixture
def data(size=100):
    isource = "All work and no play makes Jack a dull boy".split()
    itagged = "O   O    O   O  O    O     name O O    O".split()

    asource = "I'm sorry Dave I’m afraid I can't do that".split()
    atagged = "O   O     name O   O      O O     O  O".split()

    corpus = {
        "words": [isource, asource] * size,
        "tags": [itagged, atagged] * size,
        "intent": ["inform", "answer"] * size,
    }
    return pd.DataFrame(corpus)


def test_preprocessor(data):
    prep = build_preprocessor()
    prep.fit(data)
