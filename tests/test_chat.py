import pytest
import pandas as pd
import itertools
from models.chat import build_preprocessor, build_model
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=128):
    return pd.DataFrame({
        "query": [
            "How are you?",
            "I am fine as well, and where do you live?",
            "Do you like ice cream?",
        ] * size,
        "target": [
            "I am fine thanks, and you?",
            "Good, I live in Lodon.",
            "No",
        ] * size

    })


@pytest.fixture
def validation(data):
    pairs = itertools.product(data["query"].unique(), data["target"].unique())
    return pd.DataFrame(list(pairs), columns=["query", "target"])


def test_dummy(data, validation, batch_size=128):
    dataset = build_preprocessor().fit_transform(data)
    data_iter = BucketIterator(dataset, batch_size=batch_size)

    assert next(iter(data_iter)).query.shape[0] == batch_size
    assert next(iter(data_iter)).target.shape[0] == batch_size

    model = build_model()
    model.fit(data)

    validation["scores"] = model.predict(validation)
    print(validation.sort_values(["query", "scores"]))
