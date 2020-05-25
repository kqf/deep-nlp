import pandas as pd


def read_dataset():
    train = pd.read_json('train.json')
    test = pd.read_json('test.json')
    return train, test


def main():
    train, test = read_dataset()


if __name__ == '__main__':
    main()
