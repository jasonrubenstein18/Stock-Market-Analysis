# old mod

import datetime
from time import time, sleep
import os
import gc
import time
import glob
import csaps
import itertools
from itertools import permutations
from itertools import chain
from datetime import datetime, timedelta, date
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, LpInteger
import cvxpy
import pstats
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import matplotlib
from matplotlib import rc
import pulp
import plotly
import plotly_express

matplotlib.use('TkAgg')
pd.options.mode.chained_assignment = None  # default='warn'

print("Number of processors: ", mp.cpu_count())

def read_data(chunksize):
    chunksize = chunksize
    sd = pd.read_csv('___.csv', chunksize=chunksize, iterator=True)
    stock_data = pd.concat(sd, ignore_index=True)
    return stock_data


stock_data = read_data(100000)

print("Most recent date on file = " + str(stock_data['Date'].max())+"\n")
print("Most ancient date on file = " + str(stock_data['Date'].min())+"\n")


ticker_options = 0
print("Type 1 for Apple and Amazon only, or 0 for all stocks" + "\n" +
      "Typing 0 may lead to increased run time on derivative calculations")
ticker_options = input()


if ticker_options == 1:
    stock_data = stock_data[(stock_data['Ticker'] == "AAPL") | (stock_data['Ticker'] == "AMZN")]
else:
    pass


# 8.93 secs
def general_fixes(df):
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    df['DateID'] = pd.factorize(df['Date'])[0]
    df['Prev_Close'] = df.groupby("Ticker")["Adj_Close"].shift(1)
    df['pct_change'] = 0
    df['pct_change'] = 100 * (df['Adj_Close'] - df['Prev_Close'])/df['Prev_Close']
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# 33.55 secs
def simple_momentum(df):
    df = general_fixes(df)
    df["12_Day_Momentum"] = 0
    df["26_Day_Momentum"] = 0
    df["270_Day_Momentum"] = 0
    df.loc[:, "12_Day_Momentum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            12) - 1)
    df.loc[:, "26_Day_Momentum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            26) - 1)
    df.loc[:, "270_Day_Momentum"] = (
            df.groupby("Ticker")["Adj_Close"].shift(1) / df.groupby("Ticker")["Adj_Close"].shift(
            261) - 1)
    return df


# 65.11 secs
def macd(df):
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


# 35.91 secs
def twelve_two_month_price(df):
    df = simple_momentum(df)
    df = macd(df)
    if df.groupby("Ticker")["Adj_Close"].shift(263) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(263)
    elif df.groupby("Ticker")["Adj_Close"].shift(262) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(262)
    elif df.groupby("Ticker")["Adj_Close"].shift(261) is not None:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(261)
    else:
        df.loc[:, "12_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(260)
    #
    if df.groupby("Ticker")["Adj_Close"].shift(46) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(46)
    elif df.groupby("Ticker")["Adj_Close"].shift(45) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(45)
    elif df.groupby("Ticker")["Adj_Close"].shift(44) is not None:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(44)
    else:
        df.loc[:, "2_month_price"] = df.groupby("Ticker")["Adj_Close"].shift(43)
    df['30_day_12_2_momentum'] = 0
    df['12_2_change'] = (df['2_month_price'] - df['12_month_price']) / df['12_month_price']
    df['12_2_momentum'] = (df['2_month_price'] - df['12_month_price'])
    df.loc[:, "30_day_12_2_momentum"] = df.groupby("Ticker")["12_2_momentum"].shift(30)
    # Add buy signal
    df['buy'] = np.where(df['pct_change'] > 0, 1, 0)
    return df


# 420.26 secs
def derivatives(df):
    df = twelve_two_month_price(df)
    # MACD
    df['MACD_d1_velo'] = df.groupby('Ticker')['MACD'].diff()
    df['MACD_d2_acc'] = df.groupby('Ticker')['MACD_d1_velo'].diff()
    df['MACD_d3_jerk'] = df.groupby('Ticker')['MACD_d2_acc'].diff()
    # 12_2 - 30 day µ
    df['12_2_velo'] = df.groupby('Ticker')['30_day_12_2_momentum'].diff()
    df['12_2_acc'] = df.groupby('Ticker')['12_2_velo'].diff()
    df['12_2_jerk'] = df.groupby('Ticker')['12_2_acc'].diff()
    # raw Adj_Close
    df['adj_close_velo'] = df.groupby('Ticker')['Adj_Close'].diff()
    df['adj_close_acc'] = df.groupby('Ticker')['adj_close_velo'].diff()
    df['adj_close_jerk'] = df.groupby('Ticker')['adj_close_acc'].diff()
    return df


working_df_use = derivatives(stock_data)

gc.collect()


del working_df_use['DateID'], working_df_use['Prev_Close'], working_df_use['12_Day_Momentum'], \
    working_df_use['12_month_price'], working_df_use['2_month_price'],\
    working_df_use['270_Day_Momentum'], working_df_use['26_Day_Momentum']


# Write file if you don't want to re-run derivatives later
working_df_use.to_csv('...csv', index=0)


# adding constraint information for integer programming
print("Enter funds constraint (ideally below $2500, only optimizing on single share basis):\n")
funds = int(input())

print("Choose indicator (MACD or 30_day_12_2_momentum)?:\n")
indicator = input()

print("Optimization Start Date:\n")
syear, smonth, sday = int(input()), int(input()), int(input())
sdate = date(syear, smonth, sday)

print("Optimization End Date:\n")
eyear, emonth, eday = int(input()), int(input()), int(input())
edate = date(eyear, emonth, eday)


daterange = pd.date_range(sdate, edate)
dates_list = list(daterange.strftime("%Y-%m-%d"))

# Memory check
# for dtype in ['float', 'int', 'object', 'datetime']:
#     selected_dtype = working_df_use.select_dtypes(include=[dtype])
#     mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
#     mean_usage_mb = mean_usage_b / 1024 ** 2
#     print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

cols = ['Date', 'GainLoss', 'OpenCost']
returns = pd.DataFrame(columns=cols)
k = 0
unique_dates = len(working_df_use[working_df_use['Date'].isin(dates_list)]['Date'].unique())
usable_df = working_df_use[(working_df_use['Date'].isin(dates_list))
                           & (pd.notnull(working_df_use[indicator]))]
# usable_df_repeated = pd.concat([usable_df]*5, ignore_index=True).reset_index()
usable_df = usable_df[["Ticker", "Date", "Open", "Adj_Close", "pct_change", "MACD", "30_day_12_2_momentum"]]

print(len(usable_df))
# print(len(usable_df_repeated))

gc.collect()

# add in price * return to get true % change of portfolio on day (% loss can't be less than -1)
for i in dates_list:
    if indicator == "MACD":
        date_only = usable_df[(usable_df['Date'] == i)
                              & (usable_df[indicator] > 0)].reset_index(drop=True)
        data = pd.concat([date_only]*5, ignore_index=True)
    else:
        date_only = usable_df[(usable_df['Date'] == i)
                              & (usable_df[indicator] > usable_df['Open'])].reset_index(drop=True)
        data = pd.concat([date_only]*5, ignore_index=True)
    if not data.empty:
        ticker = data['Ticker']
        open_price = data['Open']
        pct_change = data['pct_change']
        # row_id = data['row_id']
        metric = data[indicator]
        P = range(len(ticker))
        S = float(funds)
        prob = LpProblem("Optimal_Portfolio", LpMaximize)
        x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)
        prob += sum(metric[p] * x[p] for p in P)
        prob += sum(open_price[p] * x[p] for p in P) <= S
        prob.solve(pulp.solvers.PULP_CBC_CMD(maxSeconds=60))
        open_price = [open_price[p] for p in P if x[p].varValue]
        ticker = [ticker[p] for p in P if x[p].varValue]
        pct_change = [pct_change[p] for p in P if x[p].varValue]
        pct_change = [x / 100 for x in pct_change]
        gain_loss = [a * b for a, b in zip(open_price, pct_change)]
        total_yield = np.sum(gain_loss)
        total_open_price = np.sum(open_price)
        returns = returns.append({
            'Date': i,
            'GainLoss': total_yield,
            'OpenCost': total_open_price
            # 'PortfolioValue': returns['PortfolioValue'].shift(1) + returns['PortfolioValue'].shift(1) * (total/100)},
            },
            ignore_index=True)
        k += 1
        print('{} out of {} dates optimized'.format(k, unique_dates))
    else:
        pass


fig = plotly_express.scatter(returns, x="Date", y="GainLoss", )
fig.show()

# print(sum(returns['Return']))

# twelve_two_return = returns
# twelve_two_return['cumulative_return'] = 0
# twelve_two_return['cumulative_return'][0] = 2500

# twelve_two_return['new_funds'] = twelve_two_return.cumulative_return.shift(1) * (1+twelve_two_return.Return.shift(1))

# macd_returns = returns
