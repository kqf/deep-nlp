import torch
import pytest
import pandas as pd


from torchtext.data import BucketIterator
from sklearn.metrics import f1_score

from models.bert import build_preprocessor, build_model, timer
from models.bert import print_mode_size


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
    with timer("Predict normal"):
        f1 = f1_score(data["is_duplicate"], model.predict(data))
    print(f"F1 score: {f1}")
    assert f1 > 0.9

    print("Model size before:")
    print_mode_size(model)

    q8bert = torch.quantization.quantize_dynamic(
        model[-1].module_._bert, {torch.nn.Linear}, dtype=torch.qint8
    )

    model[-1].module_._bert = q8bert

    print("Model size after:")
    print_mode_size(model)

    with timer("Predict quantized"):
        f1 = f1_score(data["is_duplicate"], model.predict(data))
    print(f"F1 score after quantization: {f1}")
    assert f1 > 0.9
