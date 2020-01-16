# Make sure that you have all these libraries available to run the code successfully
# Remember to update path in row 27; run from command line
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import array
import datetime as dt
import urllib.request, json
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend
from tensorflow.python.framework import ops
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Softmax, Concatenate, LSTM, Dropout, CuDNNGRU, CuDNNLSTM
from keras import layers, models
import random
from time import time, sleep
ops.reset_default_graph()
os.environ["MODIN_ENGINE"] = "dask"
import modin.pandas as pd_modin
ops.reset_default_graph()

def read_data(chunksize):
    chunksize = chunksize
    sd = pd_modin.read_csv('...', chunksize=chunksize, iterator=True)
    stock_data = pd_modin.concat(sd, ignore_index=True)
    return stock_data

stock_data = read_data(100000)

random.seed(3)
print("Choose ticker for LSTMNet:\n")
ticker = str(input())

def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def LSTMnet(ticker):
    steps = 3
    features = 1
    activation_function = 'relu'
    optimizer = 'adam'
    loss_function = 'mse'
    batch_size = 5
    epochs = 100
    verbose = 1

    X, y = list(), list()
    # for ticker in tickers:
    # scale data

    raw_seq = working_df_use[(working_df_use['Ticker'] == ticker)].reset_index(drop=True)['Adj_Close']
    # split sample
    X, y = split_sequences(raw_seq, steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], features))

    # Combined model
    model = Sequential()
    model.add(LSTM(120,activation=activation_function, input_shape=(steps, features)))
    model.add(Dense(1))
    # keras.layers.GRU(2)
    model.compile(optimizer=optimizer, loss=loss_function)

    # fit model
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # prediction
    x_input = array([raw_seq[raw_seq.idxmax()-5], raw_seq[raw_seq.idxmax()-4], raw_seq[raw_seq.idxmax()-3]])
    x_input = x_input.reshape((1, steps, features))
    yhat = model.predict(x_input, verbose=1)
    output = []
    real = []
    output.append(str(ticker) + " predicted = " + str(yhat))
    real.append("Actual price = " + str(raw_seq[raw_seq.idxmax() - 2]))
    return print(output), print(real)

LSTMnet(ticker)
