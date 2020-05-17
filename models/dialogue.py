import os
import pandas as pd


"""
!mkdir -p data
!git clone https://github.com/MiuLab/SlotGated-SLU.git
!mv SlotGated-SLU data
"""


def read_single(path):
    with open(os.path.join(path, 'seq.in')) as fwords, \
            open(os.path.join(path, 'seq.out')) as ftags, \
            open(os.path.join(path, 'label')) as fintents:

        df = pd.DataFrame({
            "words": [w.strip().split() for w in fwords],
            "tags": [t.strip().split() for t in ftags],
            "intent": [i.strip() for i in fintents],
        })
    return df


def data():
    return (
        read_single("data/SlotGated-SLU/data/atis/train/"),
        read_single("data/SlotGated-SLU/data/atis/test/"),
        read_single("data/SlotGated-SLU/data/atis/valid/"),
    )


def main():
    train, test, valid = data()
    print(train.head())


if __name__ == '__main__':
    main()
