import pandas as pd


def data():
    fname = "data/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    return pd.read_csv(fname)


def main():
    df = data()
    print(df.head())
    print(df.info())


if __name__ == '__main__':
    main()
