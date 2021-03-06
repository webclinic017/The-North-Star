import pandas as pd
import csv
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


def init():
    global ticker
    ticker_array = []
    f = open('biotechStocks.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        ticker_array.append(row[0])
        length = len(ticker_array)

    for i in range(length):
        tt = ticker_array[i]
        ticker = "{}".format(tt)
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        data = pdr.get_data_yahoo(ticker, period = "6mo", interval = "1d", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
        data.to_csv('dailyfilesDump/' + ticker + '.csv')
        
init()