import datetime as dt
import json
import pandas as pd
from dhooks import Webhook, File
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
        DayLocator, MONDAY
import pylab
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from collections import Counter
import matplotlib
import pylab
import os
import sys
import time
import csv
from matplotlib import pyplot as plt
import yahoo_fin.stock_info as si
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


# ticker_array = []
# i=0
# f = open('stocks.csv')
# csv_f = csv.reader(f)
# for row in csv_f:
#     ticker_array.append(row[0])
#     length = len(ticker_array)

# for i in range(length):
#     tt = ticker_array[i]
#     ticker = "{}".format(tt)
#     data = pdr.get_data_yahoo(ticker, start="2020-07-10", end="2020-07-28", interval='1h')
#     data.to_csv('testFiles/' + ticker + '.csv')





def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


def movingaverage(values, window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas  # as a numpy array

def weekday_candlestick(ohlc_data, ax, closep, openp, sma10, date, SP, df, fmt='%b %d', freq=7, **kwargs):
    """ Wrapper function for matplotlib.finance.candlestick_ohlc
        that artificially spaces data to avoid gaps from weekends """
    stock=('AAPL')
    # Convert data to numpy array
    ohlc_data_arr = np.array(ohlc_data)
    ohlc_data_arr2 = np.hstack(
        [np.arange(ohlc_data_arr[:,0].size)[:,np.newaxis], ohlc_data_arr[:,1:]])
    ndays = ohlc_data_arr2[:,0]  # array([0, 1, 2, ... n-2, n-1, n])
    #print(ohlc_data_arr)

    # Convert matplotlib date numbers to strings based on `fmt`
    dates = mdates.num2date(ohlc_data_arr[:,0])
    date_strings = []
    for date in dates:
        date_strings.append(date.strftime(fmt))
    
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)

    # Format x axis
    
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.tick_params(axis='y', colors='#07000d')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='#07000d')
    

    row_count = len(ndays)
    df['signal'] = 0.0
    signal = df['signal']
    counter = False
    for i in range(9, row_count):
        if closep[i] > closep[i-1] and closep[i] > closep[i-2] and closep[i] > closep[i-3]:
            print('1st:' + str(closep[i]))
            print('2nd:' + str(closep[i-1]))
            print('3rd:' + str(closep[i-2]))
            print('4th:' + str(closep[i-3]))
            df['signal'][i] = 1.0
            

            #df['signal'][i+1]
            print(closep[i])

        elif openp[i] < openp[i-1] and openp[i] < openp[i-2] and openp[i] < openp[i-3]:
            df['signal'][i] = -1.0
            print('openp1st:' + str(openp[i]))
            print('openp2nd:' + str(openp[i-1]))
            print('openp3rd:' + str(openp[i-2]))
            print('openp4th:' + str(openp[i-3]))

        if df['signal'][i] == 1.0:
            if df['signal'][i-1] == 1.0 or df['signal'][i-2] == 1.0 or df['signal'][i-3] == 1.0:
                df['signal'][i] = 0.0

        if df['signal'][i] == -1.0:
            if df['signal'][i-1] == -1.0 or df['signal'][i-2] == -1.0 or df['signal'][i-3] == -1.0:
                df['signal'][i] = 0.0

    #df['signal'] = df['signal'].shift(-4)
    df['positions'] = df['signal'].diff()
    
    #print(df['positions'])
    print(df['signal'])
    # df['sma'] = sma10
    #ax.plot(ndays, sma10, '#07000d',label='10 SMA', linewidth=1)
    ax.plot(df.loc[df.signal == 1.0].index,
         closep[df.signal == 1.0],
         '^', markersize=10, color='m')
    
    ax.plot(df.loc[df.signal == -1.0].index, 
        openp[df.signal == -1.0],
        'v', markersize=10, color='m')
    ax.set_xticks(ndays[::freq])
    ax.set_xticklabels(date_strings[::freq], rotation=45, ha='right')
    ax.set_xlim(ndays.min(), ndays.max())
    plt.ylabel('Stock price')
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')
    plt.show()
    

def buy_sell_hold():
    stock = ('AAPL')
    df = pd.read_csv('testFiles/' + stock + '.csv')
    df.rename(columns={'date': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
    df ['Date'] = df['Date'].str.replace('T', ' ', regex=True)
    df ['Date'] = df['Date'].str.replace('Z', '', regex=True)
    #df ['Date'] = df['Date'].map(lambda x: str(x)[:-9])
    #df.index.name = 'Date'
    # df.drop_duplicates(subset ="Close", 
    #                  keep = False, inplace = True)
    # df =df[df['Close'] !=0]

    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].apply(mdates.date2num)
    df = df.astype(float)
    date = df['Date']
    closep = df['Close']
    highp = df['High']
    lowp = df['Low']
    openp = df['Open']
    index = df.index
    
    date = df['Date']

    rsi = rsiFunc(closep)
    SP = len(date[5-1:])
    sma10 = movingaverage(closep, 5)
    #print(closep)
    
    x = 0
    y = len(date)
    newAr = []
    while x < y:
        appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
        newAr.append(appendLine)
        x += 1
    
    




            
    
    
    
    # for i in range(5, row_count):
    #     if (df['signal'][i] == 1.0 and df['signal'][i] == df['signal'][i-1]):
    #         df['signal'][i] == 0.0

    # for i in range(5, row_count):
    #     if (df['signal'][i] == -1.0 and df['signal'][i] == df['signal'][i-1]):
    #         df['signal'][i] == 0.0

    
    
    cond = {'a': 1, 'b': -1}
    fig = plt.figure()
    ax1 = fig.add_subplot(111,  ylabel='Price in $')
    
    
    ax1.grid(False, color='#07000d')
    
    
    
    weekday_candlestick(newAr, ax1, closep, openp, sma10, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0)
    
    plt.setp(ax1.get_xticklabels(), visible=True)
    for label in ax1.xaxis.get_ticklabels():
                label.set_rotation(45)
    # ax1.plot(df.loc[df.positions == 1.0].index, 
    #      df.sma[df.positions == 1.0],
    #      '^', markersize=10, color='m')
    
    
    
    # print(closep.tail(1))
    # print(closep.tail(2))
    # print(closep.tail(3))


buy_sell_hold()