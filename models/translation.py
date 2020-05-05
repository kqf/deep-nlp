import pandas as pd

"""
!curl http://www.manythings.org/anki/rus-eng.zip -o data/rus-eng.zip
!
!pip install pandas torch, torchtext
"""


def data():
    return pd.read_table("data/rus.txt", names=["source", "target", "caption"])


def main():
    df = data()
    print(df.head())


if __name__ == '__main__':
    main()
