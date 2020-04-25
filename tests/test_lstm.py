import pytest
import pandas as pd
from sklearn.metrics import f1_score
from models.lstm import Tokenizer, build_model


@pytest.fixture
def data(size=1000):
    return pd.DataFrame({
        "names": ["smith", "aardwark", "chair"] * size,
        "labels": [1, 0, 0] * size
    })


def test_converts_data(data):
    tokenizer = Tokenizer().fit(data["names"])
    assert tokenizer.max_len == 8

    tokenized = tokenizer.transform(data["names"])
    assert tokenized.shape, (data.shape, tokenizer.max_len)


def test_surname_classifier(data):
    model = build_model()
    model.fit(data["names"], data["labels"])

    assert f1_score(data["labels"], model.predict(data["names"])) > 0.95
