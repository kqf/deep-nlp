import pandas as pd


def read_data(filename):
    with open(filename) as f:
        data = f.readlines()
    return pd.DataFrame(
        list(zip(data[:-1], data[1:])),
        columns=["query", "target"])


def data():
    return (
        read_data("data/train.txt"),
        read_data("data/test.txt"),
        read_data("data/valid.txt"),
    )


def main():
    train, test, val = data()
    print(train.head())


if __name__ == '__main__':
    main()
