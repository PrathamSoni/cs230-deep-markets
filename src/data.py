import h5py as h5
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from collections import Counter
import tensorflow as tf
import tensorflow_hub as hub
from absl import logging

logging.set_verbosity(logging.ERROR)

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
        df.drop(columns=['sp500', "label"], inplace=True)
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
    return np.expand_dims(train_X, 3), train_y, np.expand_dims(val_X, 3), val_y, np.expand_dims(test_X, 3), test_y


def split_sp():
    _, labels = load_sp_h5()
    indices = list(range(labels.shape[0]))
    eval_size = int(.2 * len(indices))
    test = random.sample(indices, eval_size)
    indices = list(Counter(indices) - Counter(test))
    val = random.sample(indices, eval_size)
    indices = list(Counter(indices) - Counter(val))

    with open('../data/splits/train.txt', "w") as f:
        for i in indices:
            f.write(str(i) + "\n")

    with open('../data/splits/val.txt', "w") as f:
        for i in val:
            f.write(str(i) + "\n")

    with open('../data/splits/test.txt', "w") as f:
        for i in test:
            f.write(str(i) + "\n")


def get_encoder():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    # model.signatures['default'](['my text', 'batch'])
    return model


def newsh5():
    data = pd.read_csv('../data/Combined_News_DJIA.csv')
    data.drop(columns=["Date", "Label"], inplace=True)
    data = data.astype(str)
    data = data.agg("\n".join, axis=1)

    encoder = get_encoder()
    data = data.tolist()
    raw = encoder(data)
    with h5.File('../data/news.h5', 'w') as h:
        data = np.zeros((raw.shape[0] - time_len + 1, time_len, raw.shape[1]))
        for i in tqdm(range(time_len - 1, raw.shape[0])):
            # print(df[i-time_len+1:i+1].to_numpy())
            data[i - time_len + 1] = raw[i - time_len + 1:i + 1]
        h.create_dataset('data', data=data)


def get_news():
    data = h5.File('../data/news.h5', 'r')['data']
    with open('../data/splits/train.txt', "r") as f:
        train = [int(line.strip()) for line in f]
    with open('../data/splits/val.txt', "r") as f:
        val = [int(line.strip()) for line in f]
    with open('../data/splits/test.txt', "r") as f:
        test = [int(line.strip()) for line in f]

    train.sort()
    val.sort()
    test.sort()
    train_X = data[train]
    val_X = data[val]
    test_X = data[test]
    return train_X, val_X, test_X

if __name__ == '__main__':
    # newsh5()
    print(get_news())
    # sph5()
    # split_sp()
    print(get_sp())
    print("completed.")
