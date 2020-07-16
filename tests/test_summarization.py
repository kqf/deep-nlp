import pytest
import pandas as pd

from models.summarization import build_preprocessor
from models.summarization import build_model


@pytest.fixture
def data(size=100):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target": ["Work makes Jack a dull boy"] * size,
    }
    return pd.DataFrame(corpus)


def test_textpreprocessor(data):
    tp = build_preprocessor().fit(data)
    assert tp.transform(data) is not None


def test_summarizes(data):
    model = build_model().fit(data)
    print(model.transform(data.head()))
