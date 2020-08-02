import pandas as pd
import json
from tiingo import TiingoClient
import csv
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


def init():
    global ticker
    ticker_array = []
    f = open('stocks.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        ticker_array.append(row[0])
        length = len(ticker_array)
        
    print(ticker_array)

    for i in range(length):
        tt = ticker_array[i]
        ticker = "{}".format(tt)
        data = pdr.get_data_yahoo(ticker, start="2020-06-01", end="2020-07-31", interval = "1h", prepost = True)
        data.to_csv('hourDump/' + ticker + '.csv')
        
init()