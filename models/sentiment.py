import pandas as pd
from sklearn.model_selection import train_test_split


def data():
    fname = "data/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    return pd.read_csv(fname)


def main():
    df = data()
    train, test = train_test_split(df, test_size=0.5)

    print(train.head())
    print(train.info())


if __name__ == '__main__':
    main()
