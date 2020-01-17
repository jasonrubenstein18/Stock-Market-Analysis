"""
Created on Mon Jan 14 12:50:56 2019

@author: jasonrubenstein

command line program to pull stock data from yahoo finance and store as csv

remember to update paths in line 37 and 77
"""

print('This program will read in data from Yahoo Finance and store a sizable csv on your computer' + '\n' +
      'It will take between 30 and 55 mins to run')


import datetime
from time import time, sleep
import os
import pandas as pd
from pandas import DataFrame, HDFStore
import yfinance as yf
import numpy as np
from numpy import savetxt
import glob
import itertools
from itertools import permutations, chain
import datetime as dt
import gc
from concurrent import futures
# import fix_yahoo_finance as yf
# import timeit
# from datetime import datetime
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


path = '.../*.txt'
stock_files = glob.glob(path)

def get_txt(files):
    symbols = pd.concat([
                         pd.read_csv(symbols, dtype=str, error_bad_lines=False, delimiter='\t')
                         for symbols in files], axis=0)
    df = symbols.drop_duplicates(keep='first').reset_index()
    return df

stocks_raw = get_txt(stock_files)
tickers = stocks_raw['Symbol']
tickers = tickers.sort_values().reset_index(drop=True)

stocks_start = datetime.datetime(1974, 1, 1)
stocks_end = datetime.datetime(2019, 12, 20)

def get_stock_data(tickers, startdate, enddate):
    def data(ticker):
        for tickers in ticker:
            try:
                return (yf.download(ticker, start=startdate, end=enddate))
            except (ValueError, KeyError):
                pass
            time.sleep(1)
    datas = map(data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']).reset_index())


t1 = tickers

start_time = time()
#
stock_data = get_stock_data(t1, stocks_start, stocks_end)#.reset_index()


"""
Create sqlite3 database for data storage?
"""

stock_data.to_csv('...', index=0)


end_time = time()
elapsed_time = float(end_time - start_time)

print("time elapsed (in seconds): " + str(round(elapsed_time, 2)))
print("time elapsed (in minutes): " + str(round(elapsed_time / 60.0, 3)))
print('# of Rows = ' + str(len(stock_data)))
print('Min Date = ' + str(min(stock_data['Date'])))
print('Max Date = ' + str(max(stock_data['Date'])))
print('# of Tickers = ' + str(len(stock_data['Ticker'].unique())))
