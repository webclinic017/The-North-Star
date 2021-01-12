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
hook = Webhook("https://discordapp.com/api/webhooks/798367085573963796/9mLhWLjoqMicYjqUPinvj7kh7kzQtX3OZ8SdlvhujHDa5cLf_qsOmuDMN9I3IgOvwVJ8")
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
    

def weekday_candlestick(stock, ohlc_data, waveTrend, closep, openp, direction, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
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
        (6, 4), (0, 0), rowspan=4, colspan=4, facecolor='#07000d')
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)
    rsi = rsiFunc(closep)

    Label1 = '20 SMA'
    ax.plot(ndays[19:], Av1, '#FFFFFF',label=Label1, linewidth=1)
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price')
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    
    count = len(ndays) - 20
    day_labels = count / 9
    day_labels = int(round(day_labels))
    
    ax.set_xticks(ndays)
    ax.set_xlim(19, ndays.max()+1)
    ax.set_xticklabels(date_strings[19::day_labels], rotation=45, ha='right')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')





    ax2 = plt.subplot2grid(
        (6, 4), (4, 0), sharex=ax, rowspan=2, colspan=4, facecolor='#07000d')
    ax2.plot(ndays[21:], waveTrend['WT1.'][21:], '#5998ff', label='VI-', linewidth=.5)
    ax2.plot(ndays[21:], waveTrend['WT2.'][21:], '#71FA1D', label='VI+', linewidth=.5)
    fillcolor = '#00ffe8'
    ax2.fill_between(ndays[21:], waveTrend['WT1.'][21:]-waveTrend['WT2.'][21:], 0,
                        alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    ax2.axhline(y=60, linewidth=1, color='r')
    ax2.axhline(y=-60, linewidth=1, color='g')
  
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.set_xlim(19, ndays.max()+1)
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('WaveTrend', color='w')
    ax2.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    

    plt.suptitle(stock.upper(), color='w')

    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.set_yticklabels([])

    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    
    #print('Hit' + stock)
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    #plt.show()
    
    
    fig.savefig('hourPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('hourPics/' + stock + '.png')
    hook.send("SMA10-WT ALERT: " + '**' + stock + '**' + '  |  Frequency: 1 Hour' + '\n' + 'Direction: ' + direction, file=discord_pic)
    plt.close(fig)
    
    


    
   
    
    #plt.show()

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('hourfilesDump/' + stock + '.csv')
        #df = df.reset_index()
        #df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(mdates.date2num)
        #df = df.astype(float)

        del df['Adj Close']
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        volume = df['Volume']

        df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        waveTrend = TA.WTO(df)
        sma20 = movingaverage(closep, 20)

        #print(buysell)
        if (abs(waveTrend['WT1.'].iloc[-1] - waveTrend['WT2.'].iloc[-1] < .001)) and (closep.iloc[-1] < sma20[-1] and closep.iloc[-2] > sma20[-2]):
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
            
            weekday_candlestick(stock, newAr, waveTrend, closep, openp, direction, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            
        if (abs(waveTrend['WT1.'].iloc[-1] - waveTrend['WT2.'].iloc[-1] < .001)) and (closep.iloc[-1] > sma20[-1] and closep.iloc[-2] < sma20[-2]) :
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
            
            weekday_candlestick(stock, newAr, waveTrend, closep, openp, direction, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            
    except Exception as e:
        print('main loop', str(e))


for n in range(length):
    word = ticker_array[n]
    graphData(word,20,50)