import torch
import pytest
import pandas as pd

from models.dialogue import build_preprocessor, build_model
from models.dialogue import IntentClassifierModel, TaggerModel


@pytest.fixture
def data(size=100):
    isource = "All work and no play makes Jack a dull boy".split()
    itagged = "O   O    O   O  O    O     name O O    O".split()

    asource = "I'm sorry Dave I’m afraid I can't do that".split()
    atagged = "O   O     name O   O      O O     O  O".split()

    corpus = {
        "tokens": [isource, asource] * size,
        "tags": [itagged, atagged] * size,
        "intent": ["inform", "answer"] * size,
    }
    return pd.DataFrame(corpus)


def test_preprocessor(data):
    prep = build_preprocessor()
    prep.fit(data)


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_size", [100])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("intents", [2])
def test_intent_classifier_model(batch_size, seq_size, vocab_size, intents):
    batch = torch.randint(0, vocab_size, (seq_size, batch_size))
    model = IntentClassifierModel(vocab_size, intents)
    assert model(batch).shape == (batch_size, intents)


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_size", [100])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("tags_count", [100])
def test_ner(batch_size, seq_size, vocab_size, tags_count):
    batch = torch.randint(0, vocab_size, (seq_size, batch_size))
    model = TaggerModel(vocab_size, tags_count)
    assert model(batch).shape == (seq_size, batch_size, tags_count)


@pytest.mark.parametrize("modelname", ["intent", "tagger"])
def test_intent_classifier(data, modelname):
    model = build_model(modelname=modelname, epochs_count=2)
    model.fit(data, None)
