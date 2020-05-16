import pytest
import pandas as pd

from models.summarization import TextPreprocessor
from models.summarization import build_model


@pytest.fixture
def data(size=100):
    corpus = {
        "source": ["All work and no play makes Jack a dull boy"] * size,
        "target": ["Work makes Jack a dull boy"] * size,
    }
    return pd.DataFrame(corpus)


def test_textpreprocessor(data):
    tp = TextPreprocessor().fit(data)
    assert tp.transform(data) is not None


def test_translates(data):
    model = build_model(epochs_count=2)
    # First fit the text pipeline
    text = model[0]
    text.fit(data, None)
    # Then use to initialize the model
    model[-1].model_init(vocab_size=len(text[-1].text.vocab))
    # Now we are able to generate from the untrained model
    print("Before training")
    print(model.transform(data.head()))

    model.fit(data, None)
    print("After training")
    print(model.transform(data.head()))
