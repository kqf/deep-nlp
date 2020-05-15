import pandas as pd


"""
!curl -k -L "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab" -o data/news.zip
!pip install torch
!pip install torchtext
!pip install sacremoses
"""  # noqa


def data():
    return pd.read_csv("data/news.zip")


def main():
    df = data()
    print(df.head())


if __name__ == '__main__':
    main()
