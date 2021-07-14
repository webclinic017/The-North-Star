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


# now = dt.datetime.now()
# if now.hour > 16:
#     quit()
#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/786002768753328168/qmtFNOiFEA4hYoh8C6jQ6Yxof48owbR9EMgCfPBV0c5ABjyW0jzxtWlHGww2PbfBMprE")
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


def weekday_candlestick(stock, ohlc_data, closep, openp, volume, low, high, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
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
        (6, 4), (0, 0), rowspan=6, colspan=4, facecolor='#07000d')
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)

    count = len(ndays) - 10
    day_labels = count / 9
    day_labels = int(round(day_labels))
    
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.set_xticks(ndays)
    ax.set_xlim(10, ndays.max()+1)
    ax.set_xticklabels(date_strings[10::day_labels], rotation=45, ha='right')

    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

    
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    ax.axhline(low,xmin=0,xmax=ndays.max()+1, color='w', linewidth=0.5, label='Current Day Low')
    ax.axhline(high,xmin=0,xmax=ndays.max()+1, color='w', linewidth=0.5, label='Current Day High')
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    ax.tick_params(axis='x', colors='w')
   
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    plt.suptitle(stock.upper(), color='w')
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')

    volumeMin = 0
    ax1v = ax.twinx()
    ax1v.fill_between(ndays, volumeMin, volume, facecolor='#00ffe8', alpha=.4)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*volume.max())
    ax1v.set_xlim(49, ndays.max()+1)
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')

    plt.ylabel('Stock price')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=True)
    
    
    fig.savefig('dailyPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('dailyPics/' + stock + '.png')
    hook.send('Outside Day Alert: ' + '**' + stock + '**' +  '  Frequency: Daily', file=discord_pic)
    plt.close(fig)
    

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('dailyfilesDump/' + stock + '.csv')
        #df = df.reset_index()
        #df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df['Date'] = pd.to_datetime(df.index)
        df['Date'] = df['Date'].apply(mdates.date2num)
        #df = df.astype(float)

        del df['Adj Close']       
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        volume = df['Volume']
        dfv = df
        #del df['Volume']
        dfv.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)



        
        low = lowp.iat[-1]
        high = highp.iat[-1]
        #print(squeeze)

        if highp.iat[-1] > highp.iat[-2] and lowp.iat[-1] < lowp.iat[-2]:

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
            
            weekday_candlestick(stock, newAr, closep, openp, volume, low, high, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0)


            
    except Exception as e:
        print('main loop', str(e))


for n in range(length):
    word = ticker_array[n]
    graphData(word,10,50)