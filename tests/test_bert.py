import pytest
import pandas as pd


from torchtext.data import BucketIterator
from sklearn.metrics import f1_score

from models.bert import build_preprocessor, build_model


@pytest.fixture
def data(size=640):
    dataset = {
        "question1": ["How are you?", "Where are you?"] * size,
        "question2": ["How do you do?", "Where is Australia?"] * size,
        "is_duplicate": [1, 0] * size,
    }
    return pd.DataFrame(dataset)


def test_preprocessing(data, batch_size=64):
    dataset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dataset, batch_size)))

    assert batch.question1.shape[0] == batch_size
    assert batch.question2.shape[0] == batch_size
    assert batch.is_duplicate.shape == (batch_size,)


def test_model(data, batch_size=64):
    model = build_model().fit(data)
    f1 = f1_score(data["is_duplicate"], model.predict(data))
    print(f"F1 score: {f1}")
