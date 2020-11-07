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
import time
import csv
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/771576192671154186/FJ90pGS6_ocvoq31dWkOSbRJOAItpof9YIok9Qro6gA-lRi_BmI5spwhEdylVH_VkHX-")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)

matplotlib.rcParams.update({'font.size': 9})

ticker_array = []
f = open('heatmapstocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

ticker_names=['ENERGY','FINANCIAL', 'UTILITIES', 'INDUSTRIAL',
        'GOLD MINERS', 'TECH', 'HEALTH CARE', 'CONSUMER DISCRETIONARY',
                 'CONSUMER STAPLES', 'MATERIALS', 'OIL & GAS ', 'U.S. REAL ESTATE',
                  'HOMEBUILDERS', 'CONSTRUCTION', 'REAL ESTATE INDEX FUND',
                   'JUNIOR GOLD MINERS', 'ENERGY', 'OIL', 'METALS & MINING', 'RETAIL', 'SEMICONDUCTOR',
                    'BIOTECH', 'BANK', 'REGIONAL BANKING', 'TELECOM']

ticker_name = []            
for i in range(len(ticker_names)):
        ticker_name.append(ticker_names[i])


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
   

def weekday_candlestick(stock, ohlc_data, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
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
    #candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)
    rsi = rsiFunc(closep)

    Label1 = '10 SMA'
    ax.plot(ndays[9:], Av1, '#FFFFFF',label=Label1, linewidth=1)
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price')
    
    
    count = len(ndays) - 9
    day_labels = count / 9
    day_labels = int(round(day_labels))


    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.set_xticks(ndays)
    ax.set_xlim(9, ndays.max())
    ax.set_xticklabels(date_strings[9::day_labels], rotation=45, ha='right')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    plt.gcf().autofmt_xdate()
    plt.suptitle(stock.upper(), color='w')
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)
    #ax.plot(ndays,closep, linewidth=1)
    
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    
    
    

    
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    
    #print('Hit' + stock)
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    #plt.show()
    
    
    fig.savefig('hourPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('hourPics/' + stock + '.png')
    hook.send(file=discord_pic)
    plt.close(fig)

def graphData(stock, sector, MA1, MA2):
    try:
        df = pdr.get_data_yahoo(stock, period = "1mo", interval = "1h", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
        #print(df)
        #df.reindex(df.index)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df['Date'] = df.index.values
        #df = df.set_index(['Date'])
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

        close = round(closep[-1], 2)
        rsi = rsiFunc(closep)

        if rsi[-1] < 30:
            hook.send('--')
            hook.send('**'+str(sector)+'**' + '   |   Undervalued')
            now =  dt.datetime.now()
            time = now.strftime('%b %d %Y %I:%M%p')
            hook.send('Current Price: ' + str(close) + '   |   Current Time: ' + str(time))
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

            
            weekday_candlestick(stock, newAr, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            
        elif rsi[-1] > 70:
            hook.send('--')
            hook.send('**'+str(sector)+'**' + '   |   Overvalued')
            now =  dt.datetime.now()
            time = now.strftime('%b %d %Y %I:%M%p')
            hook.send('Current Price: ' + str(close) + '   |   Current Time: ' + str(time))
            
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

            
            weekday_candlestick(stock, newAr, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0) 
            

    except IndexError as e:
        print('main loop', str(e))


for n in range(length):
    word = ticker_array[n]
    print(ticker_name[n])
    sector = ticker_name[n]
    graphData(word,sector,10,50)
# for n in range(length):
#     try:
#         word = ticker_array[n]
#         os.remove('hourfilesDump/' + word + '.csv')
#     except FileNotFoundError:
#         print('File not found')