# Make sure that you have all these libraries available to run the code successfully
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend
from tensorflow.python.framework import ops
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Softmax, Concatenate, LSTM, Dropout, CuDNNGRU
from keras.layers import  CuDNNLSTM
from keras import layers, models
from numpy import array
import random
from time import time, sleep
ops.reset_default_graph()

exec(open("StockAnalysis.py").read())

# AAPL = working_df_use[(working_df_use['Ticker'] == "AAPL")].reset_index(drop=True)

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

    # demonstrate prediction
    x_input = array([raw_seq[raw_seq.idxmax()-5], raw_seq[raw_seq.idxmax()-4], raw_seq[raw_seq.idxmax()-3]])
    x_input = x_input.reshape((1, steps, features))
    yhat = model.predict(x_input, verbose=1)
    output = []
    real = []
    output.append(str(ticker) + " predicted = " + str(yhat))
    real.append("Actual price = " + str(raw_seq[raw_seq.idxmax() - 2]))
    return print(output), print(real)

LSTMnet(ticker)



# for pred_idx in range(trainlen, trainlen + futureTotal):
#     running_mean = running_mean * decay + (1.0 - decay) * prediction[pred_idx - 1]
#     run_avg_predictions.append(running_mean)
#     mse_errors.append((run_avg_predictions[-1] - trainlen[pred_idx]) ** 2)
#     run_avg_x.append(date)
# print('MSE error for EMA averaging: %.5f' % (0.5 * np.mean(mse_errors)))

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

#
# apple_training_processed = AAPL[["Date", "Adj_Close"]].set_index("Date")
#
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range = (0, 1))
#
# apple_training_scaled = scaler.fit_transform(apple_training_processed)
#
# features_set = []
# labels = []
# for i in range(271, 5420):
#     features_set.append(apple_training_scaled[i-60:i, 0])
#     labels.append(apple_training_scaled[i, 0])
#
# features_set, labels = np.array(features_set), np.array(labels)
#
# features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))



