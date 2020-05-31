import pytest
import pandas as pd


@pytest.fixture
def data():
    return pd.DataFrame({
        "query": [
            "How are you?",
            "I am fine as well, and where do you live?",
            "Do you like ice cream?",
        ],
        "target:": [
            "I am fine thanks, and you?",
            "Good, I live in Lodon.",
            "No",
        ]

    })


def test_dummy(data):
    print(data)
