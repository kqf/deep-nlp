import pandas as pd
import numpy as np


def data(dataset="train"):
    df = pd.read_csv(f"data/{dataset}.csv.zip")
    df.replace(np.nan, '', regex=True, inplace=True)
    return df


def main():
    print(data())


if __name__ == '__main__':
    main()
