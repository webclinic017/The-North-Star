import datetime as dt
import json
import pandas as pd
from dhooks import Webhook, File
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
import pylab
import os
import sys
from finta import TA
import time
import csv
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/787790236591063112/IhxxzEWoG_BNpBonZm4Bo6_x5mf_AZfptAWDK_WPbS-jfDmtIpLmOze0NLTcaBUloQfG")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)

matplotlib.rcParams.update({'font.size': 9})

ticker_array = []
f = open('stocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)


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


def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow
    

def weekday_candlestick(stock, ohlc_data, closep, openp, direction, obv,obvSMA, bbands, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
    """ Wrapper function for matplotlib.finance.candlestick_ohlc
        that artificially spaces data to avoid gaps from weekends """
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
    
    fig = plt.figure(facecolor='#07000d')

    ax = plt.subplot2grid(
        (6, 4), (1, 0), rowspan=4, colspan=4, facecolor='#07000d')
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)
    rsi = rsiFunc(closep)


    ax.plot(ndays, bbands['BB_UPPER'], '#FF3431',label='Upper Band', linewidth=1.5)
    ax.plot(ndays, bbands['BB_MIDDLE'], '#E50BD1',label='20 SMA', linewidth=1)
    ax.plot(ndays, bbands['BB_LOWER'], '#71FA1D',label='Lower Band', linewidth=1.5)
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price')
    
    count = len(ndays) - 49
    day_labels = count / 9
    day_labels = int(round(day_labels))
    
    ax.set_xticks(ndays)
    ax.set_xlim(49, ndays.max()+1)
    ax.set_xticklabels(date_strings[49::day_labels], rotation=45, ha='right')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')

    volumeMin = 0

    ax0 = plt.subplot2grid(
        (6, 4), (0, 0), sharex=ax, rowspan=1, colspan=4, facecolor='#07000d')
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'
 
    ax0.plot(ndays, rsi, rsiCol, linewidth=1.5)
    ax0.axhline(50, color='g')
    ax0.fill_between(ndays, rsi, 50, where=(rsi<= 50), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(ndays, rsi, 50, where=(rsi>= 50), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([50])
    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.set_xlim(49, ndays.max()+1)
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI', color='w')


    ax1v = plt.subplot2grid(
        (6, 4), (5, 0), sharex=ax,rowspan=1, colspan=4, facecolor='#07000d')


    ax1v.plot(ndays, obv, '#c1f9f7',label='OBV', linewidth=2)
    ax1v.plot(ndays[9:], obvSMA, '#386d13',label='OBV 10 SMA', linewidth=1.5)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(obv.min(), obv.max())
    ax1v.set_xlim(12, ndays.max()+ 1)
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')
    plt.ylabel('OBV', color='w')
    ax1v.legend(loc="lower left", framealpha=0.7, prop={'size': 6},fancybox=True)

    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    

    plt.suptitle(stock.upper(), color='w')

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax1v.set_yticklabels([])
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    
    
    fig.savefig('hourPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('hourPics/' + stock + '.png')
    hook.send("RSI-BBAND-OBV ALERT: " + '**' + stock + '**' + '  |  Frequency: 1 Hour' + '\n' + 'Direction: ' + direction, file=discord_pic)
    plt.close(fig)
    
    


    
   
    
    #plt.show()

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('hourfilesDump/' + stock + '.csv')

        df.index.name = 'Date'
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(mdates.date2num)


        del df['Adj Close']
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        volume = df['Volume']

        df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        bbands = TA.BBANDS(df)
        obv = TA.OBV(df)
        obvSMA = movingaverage(obv, 10)

        
        rsi = rsiFunc(closep)

        if (rsi[-1] > 50 and closep.iloc[-1] > bbands['BB_MIDDLE'].iat[-1] and obv.iloc[-1] > obvSMA[-1]):
            x = 0
            y = len(date)
            newAr = []
            while x < y:
                appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
                newAr.append(appendLine)
                x += 1

            Av1 = movingaverage(closep, MA1)
            Av2 = movingaverage(closep, MA2)
            SP = len(date[MA2-1:])
            direction = 'Upwards'

            
            weekday_candlestick(stock, newAr, closep, openp, obv, obvSMA, bbands, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            
        if (rsi[-1] < 50 and closep.iloc[-1] < bbands['BB_MIDDLE'].iat[-1] and obv.iloc[-1] < obvSMA[-1]):
            x = 0
            y = len(date)
            newAr = []
            while x < y:
                appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
                newAr.append(appendLine)
                x += 1

            Av1 = movingaverage(closep, MA1)
            Av2 = movingaverage(closep, MA2)
            SP = len(date[MA2-1:])
            direction = 'Downwards'

            
            weekday_candlestick(stock, newAr, closep, openp, direction, obv, obvSMA, bbands, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            
    except Exception as e:
        print('main loop', str(e))


for n in range(length):
    word = ticker_array[n]
    graphData(word,10,50)