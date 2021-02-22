import os
import h5py as h5
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from collections import Counter

time_len = 10


def load_sp_raw():
    return pd.read_csv('../data/sp500.csv')


def load_sp_h5():
    file = h5.File('../data/sp.h5', 'r')
    return file["data"], file["labels"]


def sph5():
    df = load_sp_raw()[:-1]
    with h5.File('../data/sp.h5', 'w') as h:
        h.create_dataset('labels', data=df["label"][time_len - 1:].to_numpy())
        data = np.zeros((df.shape[0] - time_len + 1, time_len, df.shape[1] - 2))
        df.drop(columns=["running average", "label"], inplace=True)
        for i in tqdm(range(time_len - 1, df.shape[0])):
            # print(df[i-time_len+1:i+1].to_numpy())
            data[i - time_len + 1] = df[i - time_len + 1:i + 1].to_numpy()
        h.create_dataset('data', data=data)


def get_sp():
    data, labels = load_sp_h5()
    with open('../data/splits/train.txt', "r") as f:
        train = [line.strip() for line in f]
    with open('../data/splits/val.txt', "r") as f:
        val = [line.strip() for line in f]
    with open('../data/splits/test.txt', "r") as f:
        test = [line.strip() for line in f]

    train.sort()
    val.sort()
    test.sort()

    train_X = data[train]
    train_y = labels[train]
    val_X = data[val]
    val_y = labels[val]
    test_X = data[test]
    test_y = labels[test]
    return train_X, train_y, val_X, val_y, test_X, test_y


def split_sp():
    _, labels = load_sp_h5()
    indices = list(range(labels.shape[0]))
    eval_size = int(.2*len(indices))
    test = random.sample(indices, eval_size)
    indices = list(Counter(indices) - Counter(test))
    val = random.sample(indices, eval_size)
    indices = list(Counter(indices) - Counter(val))

    with open('../data/splits/train.txt', "w") as f:
        for i in indices:
            f.write(str(i)+"\n")

    with open('../data/splits/val.txt', "w") as f:
        for i in val:
            f.write(str(i)+"\n")

    with open('../data/splits/test.txt', "w") as f:
        for i in test:
            f.write(str(i)+"\n")

if __name__ == '__main__':
    # sph5()
    # split_sp()
    # print(get_sp())
    print("completed.")
