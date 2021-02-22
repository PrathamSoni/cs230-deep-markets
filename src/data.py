import os
import h5py as h5
import pandas as pd
from tqdm import tqdm
import numpy as np
time_len = 10


def load_sp_raw():
    return pd.read_csv('../data/sp500.csv')

def load_sp_h5():
    pass

def sph5():
    df = load_sp_raw()[:-1]
    with h5.File('../data/sp.h5', 'w') as h:
        h.create_dataset('labels', data=df["label"][time_len-1:].to_numpy())
        data = np.zeros((df.shape[0]-time_len+1, time_len, df.shape[1]-2))
        df.drop(columns=["running average", "label"], inplace=True)
        for i in tqdm(range(time_len-1, df.shape[0])):
            # print(df[i-time_len+1:i+1].to_numpy())
            data[i-time_len+1] = df[i-time_len+1:i+1].to_numpy()
        h.create_dataset('data', data=data)

def get_sp(test_prop=.2):


if __name__=='__main__':
    sph5()