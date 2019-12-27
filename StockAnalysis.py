"""
Created on Wed Dec 18 11:20:22 2019

@author: jasonrubenstein
"""

import datetime
from time import time, sleep
import os
import pandas as pd
import numpy as np
import time
import glob
import itertools
from itertools import permutations
from itertools import chain
from datetime import datetime
import csaps
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, LpInteger
import cvxpy
import pstats
import statistics
import seaborn as sns
from matplotlib import rc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gc

pd.options.mode.chained_assignment = None  # default='warn'


def read_data(chunksize):
    chunksize = chunksize
    sd = pd.read_csv('~/stock_data.csv', chunksize=chunksize, iterator=True)
    stock_data = pd.concat(sd, ignore_index=True)
    return stock_data

stock_data = read_data(100000)

print("Most recent date on file = " + str(stock_data['Date'].max())+"\n")
print("Most ancient date on file = " + str(stock_data['Date'].min())+"\n")




working_df = stock_data[(stock_data['Ticker'] == "AAPL") | (stock_data['Ticker'] == "AMZN")]


def general_fixes(df):
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    df['DateID'] = pd.factorize(df['Date'])[0]
    df['Prev_Close'] = df.groupby("Ticker")["Adj_Close"].shift(1)
    df['pct_change'] = 0
    df['pct_change'] = 100 * (df['Adj_Close'] - df['Prev_Close'])/df['Prev_Close']
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def simple_momentum(df):
    df = general_fixes(df)
    df["12_Day_Monemntum"] = 0
    df["28_Day_Monemntum"] = 0
    df["270_Day_Monemntum"] = 0
    df.loc[:, "12_Day_Monemntum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            12) - 1)
    df.loc[:, "26_Day_Monemntum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            26) - 1)
    df.loc[:, "261_Day_Monemntum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            261) - 1)
    return df

def add_macd(df):
    df['12_day_ema'] = 0
    df['26_day_ema'] = 0
    df['MACD'] = 0
    ema_12 = df.groupby("Ticker").apply(lambda x: x["Adj_Close"].ewm(span=12).mean())
    ema_26 = df.groupby("Ticker").apply(lambda x: x["Adj_Close"].ewm(span=26).mean())
    df["12_day_ema"] = ema_12.reset_index(level=0, drop=True)
    df["26_day_ema"] = ema_26.reset_index(level=0, drop=True)
    df["MACD"] = df["12_day_ema"] - df["26_day_ema"]
    MACD = df.groupby("Ticker").apply(lambda x: x["MACD"].ewm(span=9).mean())
    df["MACD"] = MACD.reset_index(level=0, drop=True)
    return df

def twelve_two_month_price(df):
    df = simple_momentum(df)
    df = add_macd(df)
    if df.groupby("Ticker")["Adj_Close"].shift(261) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(261)
    elif df.groupby("Ticker")["Adj_Close"].shift(260) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(260)
    elif df.groupby("Ticker")["Adj_Close"].shift(262) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(262)
    else:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(263)
    if df.groupby("Ticker")["Adj_Close"].shift(44) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(44)
    elif df.groupby("Ticker")["Adj_Close"].shift(45) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(45)
    elif df.groupby("Ticker")["Adj_Close"].shift(43) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(43)
    else:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(46)
    df['30_day_12_2_momentum'] = 0
    df['12_2_change'] = (df['2_month_price'] - df['12_month_price']) / df['12_month_price']
    df['12_2_momentum'] = (df['2_month_price'] - df['12_month_price'])
    time.sleep(1)
    df.loc[:, "30_day_12_2_momentum"] = df.groupby("Ticker")["12_2_momentum"].shift(30)
    # Add buy signal
    df['buy'] = np.where(df['pct_change'] > 0, 1, 0)
    return df

def derivatives(df):
    df = twelve_two_month_price(df)
    # MACD
    df['MACD_d1_velo'] = df.groupby('Ticker')['MACD'].transform(pd.Series.diff)
    df['MACD_d2_acc'] = df.groupby('Ticker')['MACD_d1_velo'].transform(pd.Series.diff)
    df['MACD_d3_jerk'] = df.groupby('Ticker')['MACD_d2_acc'].transform(pd.Series.diff)
    # 12_2 - 30 day µ
    df['12_2_velo'] = df.groupby('Ticker')['30_day_12_2_momentum'].transform(pd.Series.diff)
    df['12_2_acc'] = df.groupby('Ticker')['12_2_velo'].transform(pd.Series.diff)
    df['12_2_jerk'] = df.groupby('Ticker')['12_2_acc'].transform(pd.Series.diff)
    # raw Adj_Close
    df['adj_close_velo'] = df.groupby('Ticker')['Adj_Close'].transform(pd.Series.diff)
    df['adj_close_acc'] = df.groupby('Ticker')['adj_close_velo'].transform(pd.Series.diff)
    df['adj_close_jerk'] = df.groupby('Ticker')['adj_close_acc'].transform(pd.Series.diff)
    return df


# print("Choose your ticker\n")
# test = twelve_two_month_price(working_df[(working_df['Ticker'] == input())])
#
# test['buy'] = np.where(test['30_day_12_2_momentum'] > 0, 1, 0)
# buys = test[(test['buy'] == 1)]
#
# print(test['pct_change'].sum())


working_df_use = derivatives(working_df)
gc.collect()




print("Enter date (YYYY/MM/DD):\n")
date = input()

print("dateID = " + str(working_df[(working_df['Date'] == date)]['DateID'].unique()))

# print("Enter DateID:\n")
# dateID = input()
# dateID = int(dateID)

# dateID_end = str(int(dateID) + 3)
# dates = list(range(dateID, dateID_end))

print("Enter funds:\n")
funds = int(input())

print("What indicator (MACD or 30_day_12_2_momentum)?:\n")
indicator = input()


def optimize_portfolio(df, date, funds, indicator):
    df = df.dropna()
    data = df[(df['Date'] == date)].reset_index(drop=True)
    # date = df[(df['DateID'] == dateID)]['Date'].unique()
    ticker = data['Ticker']
    open_price = data['Open']
    dateID = data['DateID']
    pct_change = data['pct_change']
    indicator = data[indicator]
    # buy = data['buy']
    P = range(len(ticker))
    S = int(funds)
    prob = LpProblem("Optimal_Portfolio", LpMaximize)

    # Declare decision variable x, which is 1 if a
    # stock is part of the portfolio and 0 else
    x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)

    # Objective function -> Maximize momentum
    prob += sum(indicator[p] * x[p] for p in P)

    # Constraint on funds
    prob += sum(open_price[p] * x[p] for p in P) <= S

    # Solve the objective func
    # LpSolverDefault.msg = 0
    prob.solve()

    # solution; pull ticker, open_price, pct_change, and date
    portfolio = [ticker[p] for p in P if x[p].varValue]
    open_price = [open_price[p] for p in P if x[p].varValue]
    pct_change = [pct_change[p] for p in P if x[p].varValue]
    date = date

    # full = pd.DataFrame(portfolio)
    # full.columns = ['Ticker']

    # full = pd.merge(full, data, how='left', on='Ticker')
    # full.to_csv('/Users/jasonrubenstein/Downloads/optimal_portfolio.csv')
    return print([portfolio, np.round(open_price,2), np.round(pct_change,2)], date)


optimize_portfolio(working_df, date, funds, indicator)


print("Choose ticker for Echo State Network:\n")
ticker = str(input())

def esn_predict(df, ticker, trainlen):
    sparsity = 0.2
    rand_seed = 23
    spectral_radius = 1.2
    noise = .0005
    n_reservoir = 500

    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = n_reservoir,
              sparsity=sparsity,
              random_state=rand_seed,
              spectral_radius = spectral_radius,
              noise=noise)

    trainlen = 1500
    future = 30
    futureTotal=60
    pred_tot=np.zeros(futureTotal)

    test = df[(df['Ticker'] == ticker)]
    test = test.tail(trainlen+futureTotal)
    test = test.dropna()
    test = test[['Adj_Close']]
    testarray = np.squeeze(np.asarray(test)).astype('float64')

    for i in range(0, futureTotal, future):
        pred_training = esn.fit(np.ones(trainlen), testarray[i:trainlen+i])
        prediction = esn.predict(np.ones(future))
        pred_tot[i:i+future] = prediction[:, 0]

    run_avg_predictions = []
    mse_errors = []
    run_avg_x = []
    decay = 0.5
    running_mean = 0.0
    run_avg_predictions.append(running_mean)


    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=False)

    plt.figure(figsize=(16,8))
    plt.plot(range(1000,trainlen+futureTotal),testarray[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
    #plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
    plt.plot(range(trainlen,trainlen+futureTotal),pred_tot,'k',  alpha=0.8, label='Free Running ESN')

    lo,hi = plt.ylim()
    plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

    plt.title(r'Echo State Network ' + ticker + ' Price Real vs. Predicted', fontsize=25)
    plt.xlabel(r'Time (Days)', fontsize=20,labelpad=10)
    plt.ylabel(r'Price ($)', fontsize=20,labelpad=10)
    plt.legend(fontsize='x-large', loc='best')
    sns.despine()
    return 0

esn_predict(working_df_use, ticker, 2000)


# plt.close()

'''
To do:

- Loop test through multiple dates, sum pct_change for each
- Use neural network to project next days price
- Interpolate spline and find derivatives as indicators of buying opportunity

'''


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
# AMZN_plot = AMZN_test[(AMZN_test['Date'] > "2018-02-10") & (AMZN_test['Date'] < "2019-12-01")]
# plt.plot(AMZN_plot['Date'], AMZN_plot['12_2_momentum'])
# plt.show()
# plt.close()


# def pct_change_iloc(df):
#     df['pct'] = 100 * (1 - df.iloc[0].Close / df.Close)
#     return df
#
# def change_from_start(df, start_date, end_date):
#     df = df[(df['Date'] > start_date) & (df['Date'] < end_date)]
#     df_change_grp = df.groupby('Ticker').apply(pct_change_iloc)
#     # df_change_pct = df_change_grp[(df_change_grp['pct'] > percent)]
#     return df_change_grp
#

