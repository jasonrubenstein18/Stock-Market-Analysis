import datetime
import os
import gc
import time
from time import time, sleep
import glob
import csaps
import itertools
from datetime import datetime, timedelta, date
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value, LpInteger
from time import time, sleep
import pstats
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib
from matplotlib import rc
import plotly
import plotly_express
from multiprocessing import Pool


def startup():
    matplotlib.use('TkAgg')
    pd.options.mode.chained_assignment = None  # default='warn'
    print("Number of processors: ", mp.cpu_count())
    exec(open("pyESN.py").read())


def read_data(chunksize):
    cols = ['Ticker', 'Date',	'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    full_datas = pd.DataFrame(columns=cols)
    path = '/Users/jasonrubenstein/Desktop/Python/QuantFin1/StockData/*.csv'
    stock_files = glob.glob(path)
    for i in stock_files:
        sd = pd.read_csv(i, chunksize=chunksize,
                         iterator=True)
        stock_data = pd.concat(sd, ignore_index=True)
        full_datas = full_datas.append(stock_data, ignore_index=True)
    # sd = pd.read_csv('~/Desktop/Python/QuantFin1/StockData/stock_data2020-03-19.csv',
    #                  chunksize=chunksize, iterator=True)
    # stock_data = pd.concat(sd, ignore_index=True)
    return full_datas, print("Most recent date on file = " + str(full_datas['Date'].max()) + "\n")
    print("Most ancient date on file = " + str(full_datas['Date'].min()) + "\n")



stock_data = read_data(100000)



ticker_options = 0
print("Type 1 for Apple and Amazon only, or 0 for all stocks" + "\n" +
      "Typing 0 may lead to increased run time on derivative calculations")
ticker_options = input()


if ticker_options == 1:
    stock_data = stock_data[(stock_data['Ticker'] == "AAPL") | (stock_data['Ticker'] == "AMZN")]
else:
    pass



def general_fixes(df):
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    df['DateID'] = pd.factorize(df['Date'])[0]
    df['Prev_Close'] = df.groupby("Ticker")["Adj_Close"].shift(1)
    df['pct_change'] = 0
    df['pct_change'] = (df['Adj_Close'] - df['Open'])/df['Open'] * 100
    df['Prev_pct_change'] = df.groupby("Ticker")["pct_change"].shift(1)
    df['Prev_volume'] = df.groupby("Ticker")["Volume"].shift(1)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


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



def mu_pct_change(df):
    df2 = general_fixes(df)
    ema_pct_change = df.groupby("Ticker").apply(lambda x: x["Prev_pct_change"].ewm(span=100).mean())
    df2["100_day_µ_pct_change"] = ema_pct_change.reset_index(level=0, drop=True)
    ema_volume = df.groupby("Ticker").apply(lambda x: x["Prev_volume"].ewm(span=100).mean())
    df2["100_day_µ_volume"] = ema_volume.reset_index(level=0, drop=True)
    return df2

# Choose whatever indicators you want to use or functions to run
working_data_all = mu_pct_change(stock_data)

###### FUNDS ######
print("Enter funds constraint (don't make super high since max n-qty of individual stock):\n")
FUNDS = int(input())

###### INDICATOR ######
print("Choose indicator (MACD, 30_day_12_2_momentum, 100_day_µ_pct_change)?:\n")
INDICATOR = str(input())

###### START DATE ######
print("Start Date (year * month * day):\n")
SYEAR, SMONTH, SDAY = int(input()), int(input()), int(input())
sdate = date(SYEAR, SMONTH, SDAY)

###### END DATE ######
print("End Date (year * month * day):\n")
EYEAR, EMONTH, EDAY = int(input()), int(input()), int(input())
edate = date(EYEAR, EMONTH, EDAY)

daterange = pd.date_range(sdate, edate)
dates_list = list(daterange.strftime("%Y-%m-%d"))

# Memory check
# for dtype in ['float', 'int', 'object', 'datetime']:
#     selected_dtype = working_df_use.select_dtypes(include=[dtype])
#     mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
#     mean_usage_mb = mean_usage_b / 1024 ** 2
#     print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

# Russell 1000 stocks only
russ = pd.read_csv('~/Desktop/Python/QuantFin1/Russell1000Stocks.csv', sep=",")

working_df_use = working_data_all[(working_data_all['Ticker'].isin(russ['Stock']))].reset_index(drop=True)

# Define the dataframe to be appended; run from here
cols = ['Date', 'DailyGainLoss', 'OpenCost', 'CumulativeGainLoss', 'PosNeg', 'Tickers']
book = pd.DataFrame(columns=cols)

# Set iterator k equal to 0
k = 0

# number of unique dates in dataframe
unique_dates = len(working_df_use[working_df_use['Date'].isin(dates_list)]['Date'].unique())


bad_tickers = ["TWLO", "INVH", "VST", "FHB", "HGV", "ARD"]
# include only dates in desired range where the indicator is not null
usable_df = working_df_use[(working_df_use['Date'].isin(dates_list))
                           & (pd.notnull(working_df_use[indicator]))
                           & (working_df_use['100_day_µ_volume'] > 250000) # volume definition. To add VWMA
                           & (~working_df_use['Ticker'].isin(bad_tickers)).reset_index(drop=True)]

# remove unnecessary columns
# usable_df = usable_df[["Ticker", "Date", "Open", "Volume", "Adj_Close",
#                        "pct_change", "MACD", "30_day_12_2_momentum"]]

usable_df = usable_df[["Ticker", "Date", "Open", "Volume", "Adj_Close",
                       "pct_change", indicator]]
print(len(usable_df))
print(len(usable_df['Ticker'].unique()))

# remove rows with date that has already been optimized
# usable_df_new = usable_df[~usable_df['Date'].isin(book['Date'])].reset_index(drop=True)

# print(len(usable_df_new))

# works but time complexity is an issue. It's erratic not necessarily bad

for i in dates_list:
    usable_df_new = usable_df[~usable_df['Date'].isin(book['Date'])].reset_index(drop=True)
    # if indicator == "MACD":
    date_only = usable_df_new[(usable_df_new['Date'] == i)
                              & (usable_df_new[indicator] > 0)].reset_index(drop=True)
    data = pd.concat([date_only]*20, ignore_index=True).reset_index(drop=True) # max 20 shares of any individual stock per date
    # data.rename(columns={'index': 'row_id'}, inplace=True)
    # else:
    #     date_only = usable_df_new[(usable_df_new['Date'] == i)
    #                               & (usable_df_new[indicator] > 0)].reset_index(drop=True)
    #     data = pd.concat([date_only]*5, ignore_index=True).reset_index()
    if not data.empty:
        start_time = time()
        ticker, open_price, pct_change, metric = data['Ticker'], data['Open'], data['pct_change'], data[indicator]
        # open_price = data['Open']
        # pct_change = data['pct_change']
        # row_id = data['row_id']
        # metric = data[indicator]
        P = range(len(ticker))
    # add in funds determinant based upon prior few days % return (if bunch of negatives invest less of funds)
        if len(book) < 10:
            book['FundSignal'] = 0
        else:
            book['FundSignal'] = book.PosNeg.rolling(window=10).mean().sort_index(level=1).values
            # book['FundSignal'].fillna(0, inplace=True)
        if len(book) < 1:
            S = float(funds)
        elif len(book) > 9 and abs(book.FundSignal.iat[-1]) > float(0.5):
            # use half of funds available
            # book['FundSignal'].fillna(0, inplace=True)
            S = (float(funds) + float(book.CumulativeGainLoss.iat[-1]))*.5
        else:
            S = float(funds)+float(book.CumulativeGainLoss.iat[-1])

    # Moving average of last few days book, with binary pos/neg. (e.g. if µ last 5 days < -.6, use half of available capital)
        prob = LpProblem("Optimal_Portfolio", LpMaximize)
        x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)
        prob += sum(metric[p] * x[p] for p in P)
        prob += sum(open_price[p] * x[p] for p in P) <= S
        prob.solve()
        open_price = [open_price[p] for p in P if x[p].varValue]
        ticker = [ticker[p] for p in P if x[p].varValue]
        pct_change = [pct_change[p] / 100 for p in P if x[p].varValue]
        # pct_change = [x / 100 for x in pct_change]
        gain_loss = [a * b for a, b in zip(open_price, pct_change)]
        total_yield, total_open_price = np.sum(gain_loss), np.sum(open_price)
        ticker = np.array(ticker)
        # total_open_price = np.sum(open_price)
        # dates, gains, cost = set(), set(), set()
        # dates.add(i)
        # gains.add(total_yield)
        # cost.add(total_open_price)
        # gc.disable()
        book = book.append({
            'Date': i,
            'DailyGainLoss': total_yield,
            'OpenCost': total_open_price,
            'Tickers': np.unique(ticker)
            },
            ignore_index=True)
        # book['Tickers'] = book.apply(lambda r: tuple(r), axis=1).apply(ticker)
        book['CumulativeGainLoss'] = book['DailyGainLoss'].cumsum()
        book['PosNeg'] = np.where(book['DailyGainLoss'] < -10,
                                     -1,
                                     0)
        end_time = time()
        k += 1
        elapsed_time = float(end_time - start_time)
        print('{} out of {} dates optimized in {} seconds'.format(k, unique_dates, round(elapsed_time, 2)))
        print(date_only.loc[date_only[indicator].idxmax()])
        print(date_only.loc[date_only['pct_change'].idxmax()])
        # print('{} out of {} dates optimized in {} seconds'.format(k, unique_dates, round(elapsed_time, 2)))

        if elapsed_time > 10:
            try:
                sleep(15)
                gc.collect()
                continue
            except AttributeError:
                print("time.sleep attribute error occurred, passing over sleep func.")
                gc.collect()
                continue
        else:
            continue
    else:
        pass

# end run


fig = plotly_express.line(book, x="Date", y="CumulativeGainLoss")
fig.show()

book = book.fillna(0)
print("Total Return From {} to {} = ".format(sdate, edate) + str(round(sum(book['DailyGainLoss']),2)))
print("Total Return From {} to {} = ".format(sdate, edate) + str(round(sum(book['DailyGainLoss'])/funds*100,2))+"%")
