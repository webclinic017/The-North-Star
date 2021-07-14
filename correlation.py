import datetime as dt
import json
import pandas as pd
from dhooks import Webhook, File
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.patches as mpatches
import matplotlib
import pylab
import os
import sys
import time
import csv
import yfinance as yf
from pandas_datareader import data as pdr
from finta import TA
yf.pdr_override()

tickers = ['AMZN', 'AAPL', 'TSLA', 'GS', 'BAC', 'C', 'ATVI', 'EA']

returns = pd.Dataframe()

for ticker in tickers:
    data = pdr.get_data_yahoo(ticker, period = "6mo", interval = "1d", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
    data[ticker] = data['Adj Close'].pct_change()

    if returns.empty:
        returns = data[[ticker]]
    else:
        returns = returns.join(data[[ticker]], how = 'outer')

returns = returns.dropna()
correlation = returns.corr()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xticks(np.arange(correlation.shape[0])+ 0.5, minor = False)
ax.set_yticks(np.arange(correlation.shape[1])+ 0.5, minor = False)
heatmap = ax.pcolor(correlation, cmap = plt.cm.RdYlGn)
fig.colorbar(heatmap)


ax.set_xticklabels(correlation.index)
ax.set_yticklabels(correlation.columns)
plt.show()