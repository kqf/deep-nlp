import time
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


"""
!mkdir -p data
!pip3 -qq install torch
!pip install -qq bokeh
!pip install -qq eli5
!pip install scikit-learn
!curl -k -L "https://drive.google.com/uc?export=download&id=1ji7dhr9FojPeV51dDlKRERIqr3vdZfhu" -o data/surnames-multilang.txt
"""  # noqa


def data(filename="data/surnames-multilang.txt"):
    return pd.read_csv(filename, sep="\t", names=["surname", "label"])


def baseline(X_tr, X_te, y_tr, y_te):
    model = make_pipeline(
        CountVectorizer(analyzer='char', ngram_range=(1, 4)),
        LogisticRegression(),
    )
    model.fit(X_tr, y_tr)

    print("Train")
    print(classification_report(y_tr, model.predict(X_tr)))

    print("Test")
    print(classification_report(y_te, model.predict(X_te)))
    return model


def main():
    df = data()
    print(df)
    le = LabelEncoder()
    X, y = df["surname"], le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    base = baseline(X_tr, X_te, y_tr, y_te)
    print(base)


if __name__ == '__main__':
    main()
