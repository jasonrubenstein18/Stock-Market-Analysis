import datetime
from time import time, sleep
import pandas as pd
import yfinance as yf
import glob
import requests
import gc

print('This program will read in data from Yahoo Finance and store a sizable csv on your computer' + '\n' +
      'This will take some time to run')

path = '/Users/jasonrubenstein/Desktop/Python/QuantFin1/Tickers/Use/*.txt'
stock_files = glob.glob(path)


def get_txt(files):
    symbols = pd.concat([
                         pd.read_csv(symbols,
                                     dtype=str,
                                     error_bad_lines=False,
                                     delimiter='\t')
        for symbols in files], axis=0)
    df = symbols.drop_duplicates(keep='first').reset_index()
    return df


stocks_raw = get_txt(stock_files)
tickers = stocks_raw['Symbol']
tickers = tickers.sort_values().reset_index(drop=True)

stocks_start = datetime.datetime(1970, 3, 20)
# stocks_end = datetime.datetime(2020, 11, 2)
stocks_end = datetime.datetime.today()


def get_stock_data(tickers, startdate, enddate):
    def data(ticker):
        # with futures.ProcessPoolExecutor() as pool:
        for tickers in ticker:
            try:
                return (yf.download(ticker, start=startdate, end=enddate))
            except (ValueError, KeyError, requests.exceptions.ConnectionError):
                gc.collect()
                pass
            time.sleep(1)
    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker', 'Date']).reset_index()

t1 = ["SPY"]

t1 = tickers

start_time = time()
#
stock_data = get_stock_data(t1, stocks_start, stocks_end)#.reset_index()


"""
Create sqlite3 database for data storage?
"""

stock_data.to_csv('/Users/jasonrubenstein/Desktop/Python/QuantFin1/StockData/stock_data'
                  + str(stock_data['Date'].max()) + '.csv', index=0)

# stock_data.to_pickle('/Users/jasonrubenstein/Desktop/Python/QuantFin1/StockData/stock_data')

# creating a HDF5 file
# store = HDFStore('stock_data_full.h5')
# adding dataframe to the HDF5 file
# store.put('stock_data', stock_data, format='table', data_columns=True)

end_time = time()
elapsed_time = float(end_time - start_time)

print("time elapsed (in seconds): " + str(round(elapsed_time, 2)))
print("time elapsed (in minutes): " + str(round(elapsed_time / 60.0, 3)))
print('# of Rows = ' + str(len(stock_data)))
print('Min Date = ' + str(min(stock_data['Date'])))
print('Max Date = ' + str(max(stock_data['Date'])))
print('# of Tickers = ' + str(len(stock_data['Ticker'].unique())))
