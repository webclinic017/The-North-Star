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
hook = Webhook("https://discordapp.com/api/webhooks/750357294037991524/3mrrM_4RCrDY70Yhv6M6zP-eNCx8YfMntlHRwqZvDGzSTdcbKSRlpA8zQ-Fiz_VOAdHG")
# with open('lord.png', 'r+b') as f:
#     img = f.read()  # bytes

# hook.modify(name='TheMeciah', avatar=img)

matplotlib.rcParams.update({'font.size': 9})

ticker_array = []
f = open('stocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

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
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        data = pdr.get_data_yahoo(ticker, period = "1mo", interval = "1h", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
        data.to_csv('hourRSIfiles/' + ticker + '.csv')

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
    


def weekday_candlestick(stock, ohlc_data, closep, openp, chaikin, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
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

    Label1 = '10 SMA'
    Label2 = '50 SMA'
    ax.plot(ndays[9:], Av1, '#FFFFFF',label=Label1, linewidth=1)
    ax.plot(ndays[-SP:], Av2, '#FFFF00',label=Label2, linewidth=1)
    
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price')
    
    count = len(ndays) - 49
    day_labels = count / 10
    day_labels = int(round(day_labels))
    
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.set_xticks(ndays)
    ax.set_xlim(49, ndays.max())
    ax.set_xticklabels(date_strings[49::day_labels], rotation=45, ha='right')
    #print(date_strings[49::day_labels])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #plt.locator_params(axis='x', nbins=10)

    # ax.yaxis.set_major_locator(
    #     mticker.MaxNLocator(nbins=5, prune='upper'))
    
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    
    maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                        fancybox=True, borderaxespad=0.)
    maLeg.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')

    #volumeMin = 0

    # ax0 = plt.subplot2grid(
    #     (6, 4), (0, 0), sharex=ax, rowspan=1, colspan=4, facecolor='#07000d')
    # # rsi = rsiFunc(closep)
    # rsiCol = '#c1f9f7'
    # posCol = '#386d13'
    # negCol = '#8f2020'

    # ax0.plot(ndays, rsi, rsiCol, linewidth=1.5)
    # ax0.axhline(70, color=negCol)
    # ax0.axhline(30, color=posCol)
    # ax0.fill_between(ndays, rsi, 70, where=(rsi>= 70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    # ax0.fill_between(ndays, rsi, 30, where=(rsi<= 30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    # ax0.set_yticks([30, 70])
    # ax0.yaxis.label.set_color("w")
    # ax0.spines['bottom'].set_color("#5998ff")
    # ax0.spines['top'].set_color("#5998ff")
    # ax0.spines['left'].set_color("#5998ff")
    # ax0.spines['right'].set_color("#5998ff")
    # ax0.tick_params(axis='y', colors='w')
    # ax0.tick_params(axis='x', colors='w')
    # plt.ylabel('RSI')
    volumeMin = 0
    #ax1v = ax.twinx()
    ax1v = plt.subplot2grid(
        (6, 4), (4, 0), sharex=ax,rowspan=1, colspan=4, facecolor='#07000d')
    bars = ax1v.bar(ndays,volume,facecolor='g')

# Loop over bars
    for i, bar in enumerate(bars):
        if i == 0: 
            continue

        # Find bar heights
        h = bar.get_height()*2
        h0 = bars[i-1].get_height()*2
        # Change bar color to red if height less than previous bar
        if h < h0: bar.set_facecolor('r')
    #ax1v.fill_between(ndays, volumeMin, volume, facecolor='#00ffe8', alpha=.4)
    #ax1v.axes.yaxis.set_ticklabels([])
    #plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    #ax1v.tick_params(axis='x', colors='w')
    plt.ylabel('Volume')
    ax1v.grid(False)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, volume.max())
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')


    ax2 = plt.subplot2grid(
        (6, 4), (5, 0), sharex=ax, rowspan=1, colspan=4, facecolor='#07000d')
    fillcolor = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(closep)
    ema9 = ExpMovingAverage(macd, nema)
    ax2.plot(ndays, chaikin, '#FF3431', label='Chaikin', linewidth=1)
    ax2.axhline(y=0, linewidth=1, color='y')
    # VIm = mpatches.Patch(color='#4ee6fd', label= 'VI-')
    # VIm = mpatches.Patch(color='#71FA1D', label= 'VI+')
    #ax2.legend(loc="lower left", framealpha=1, prop={'size': 7},fancybox=True)
    
    # ax2.plot(ndays, macd, color='#4ee6fd', lw=2)
    # ax2.plot(ndays, ema9, color='#e1edf9', lw=1)
    # ax2.fill_between(ndays, macd-ema9, 0,
    #                     alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('Chaikin', color='w')
    ax2.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    plt.suptitle(stock.upper(), color='w')

    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)

    
    #print('Hit' + stock)
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    plt.show()
    
    
    # fig.savefig('hourPics/' + stock + '.png', facecolor=fig.get_facecolor())
    # discord_pic = File('hourPics/' + stock + '.png')
    # hook.send("Chaiking Oscillator: " + stock + "  Frequency: 1 hour", file=discord_pic)
    # plt.close(fig)
    

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('hourfilesDump/' + stock + '.csv')
   
        #df = df.reset_index()
        #df.index = pd.to_datetime(df.index)
        #print(df)

        df.index.name = 'Date'

        df['Date'] = pd.to_datetime(df.index)
        df['Date'] = df['Date'].apply(mdates.date2num)
        #df = df.astype(float)
        #print(df)
        del df['Adj Close']
        #del df['Volume']
        
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        #volume = df['Volume']
        dfv = df
        del dfv['Date']
        dfv.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        #print(dfv)
        waveTrend = TA.WTO(dfv)
        print(waveTrend)
        #print(vortex)

        rsi = rsiFunc(closep)

        #if abs(vortex['VIp'].iloc[-1] - vortex['VIm'].iloc[-1]) < .01:
            
        x = 0
        y = len(date)
        newAr = []
        while x < y:
            appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
            newAr.append(appendLine)
            x += 1
        #print(newAr)
        Av1 = movingaverage(closep, MA1)
        Av2 = movingaverage(closep, MA2)

        SP = len(date[MA2-1:])
        #print(SP)
        
        weekday_candlestick(stock, newAr, closep, openp, chaikin, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0)

    except Exception as e:
        print('main loop', str(e))


#newData()
#init()
# for n in range(length):
#     word = ticker_array[n]
#     graphData(word,10,50)
graphData('AEP', 10, 50)















