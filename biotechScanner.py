import datetime as dt
import json
import pandas as pd
from dhooks import Webhook, File, Embed
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
from tweetpull import pull
from twitter import twitter
yf.pdr_override()

hook = Webhook("https://discordapp.com/api/webhooks/798390938745569291/zLf9lp5nts64Jdau7nNdxPf3wfuhyfwq_wdJnaZeR2IqkTsJpFGVCwoHz7bBOxeMScaT")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='Biotech Scanner', avatar=img)

ticker_array = []
f = open('biotechStocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

def percentChange(startPoint, currentPoint):
    return ((currentPoint-startPoint)/startPoint)*100


def Scanner(stock):
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

        current = df.Close.iat[-1]
        last = df.Close.iat[-2]
        pct = percentChange(last, current)
        print(pct)
    
        

        if (pct > 10.0):
            data = pdr.get_data_yahoo(stock, period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
            sma7 = data['Close'].rolling(7).mean()
            x = []
            for n in range(len(data)):
                x.append(n)
                
            fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
            plt.plot(x, data['Close'], color='#07000d')
            plt.plot(x, sma7, color='#5998ff')
            plt.axis('off')
            
            fig.savefig('hourPics/' + str(stock) + '.png')
            pct = round(pct, 3)
            price = round(data['Close'][-1], 3)

            embed = Embed(
                description='Upmover :arrow_up_small:' + '\n' + '**Percent Change:** ' + str(pct),
                color=0x5CDBF0,
                timestamp='now'  # sets the timestamp to current time
                )


            image2 = 'https://i.imgur.com/f1LOr4q.png'

            embed.set_author(name=str(stock))

            embed.add_field(name='Current Price', value= '$' + str(price))
            embed.set_footer(text='Hourly Graph of the Past Week. Blue line is 7 SMA')
            discord_pic = File('hourPics/' + str(stock) + '.png', name= 'stock.png')
            embed.set_image('attachment://stock.png')
            hook.send(embed=embed, file = discord_pic)
            plt.close(fig)


        if (pct < -10.0):
            data = pdr.get_data_yahoo(stock, period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
            sma7 = data['Close'].rolling(7).mean()
            x = []
            for n in range(len(data)):
                x.append(n)
                
            fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
            plt.plot(x, data['Close'], color='#07000d')
            plt.plot(x, sma7, color='#FF0000')
            plt.axis('off')
             
            fig.savefig('hourPics/' + str(stock) + '.png')
            pct = round(pct, 3)
            price = round(data['Close'][-1], 3)
    
            embed = Embed(
                description='Downmover :small_red_triangle_down:' + '\n' + '**Percent Change:** ' + str(pct),
                color=0xff0000,
                timestamp='now'  # sets the timestamp to current time
                )


            image2 = 'https://i.imgur.com/f1LOr4q.png'

            embed.set_author(name=str(stock))

            embed.add_field(name='Current Price', value= '$' + str(price))
            embed.set_footer(text='Hourly Graph of the Past Week. Red line is 7 SMA')
            discord_pic = File('hourPics/' + str(stock) + '.png', name= 'stock.png')
            embed.set_image('attachment://stock.png')
            hook.send(embed=embed, file = discord_pic)
            plt.close(fig)
    except IndexError: 
        print('Index Error')

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


def weekday_candlestick(stock, ohlc_data, pct, price, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=50, **kwargs):
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

    ax.margins(1)

    
    ax.yaxis.label.set_color("w")
    ax.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax.tick_params(axis='x', colors='w')
    plt.ylabel('Stock price')
    
    count = len(ndays)
    day_labels = count / 9
    day_labels = int(round(day_labels))
    
    ax.spines['bottom'].set_color("#5998ff")
    ax.spines['top'].set_color("#5998ff")
    ax.spines['left'].set_color("#5998ff")
    ax.spines['right'].set_color("#5998ff")
    ax.set_xticks(ndays)
    ax.set_xlim(0, ndays.max()+ 1)
    ax.set_xticklabels(date_strings[0::day_labels], rotation=45, ha='right')
    
    #print(date_strings[49::day_labels])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #plt.locator_params(axis='x', nbins=10)

    # ax.yaxis.set_major_locator(
    #     mticker.MaxNLocator(nbins=5, prune='upper'))
    
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    
    # maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
    #                     fancybox=True, borderaxespad=0.)
    # maLeg.get_frame().set_alpha(0.4)
    # textEd = pylab.gca().get_legend().get_texts()
    # pylab.setp(textEd[0:5], color='#07000d')
    plt.suptitle(stock.upper(), color='#07000d')


    volumeMin = 0
    #ax1v = ax.twinx()
    ax1v = plt.subplot2grid(
        (6, 4), (4, 0), sharex=ax,rowspan=2, colspan=4, facecolor='#07000d')
    ax1v.margins(1)
    bars = ax1v.bar(ndays,volume,facecolor='g')

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
    plt.ylabel('Volume', color='w')
    ax1v.grid(False)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, volume.max())
    ax1v.set_xlim(0, ndays.max()+ 1)
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')
    
    ax1v.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax1v.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    plt.suptitle(stock.upper(), color='w')
    

    plt.setp(ax.get_xticklabels(), visible=False)
    #ax1v.set_yticklabels([])


    
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    fig.savefig('dailyPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('dailyPics/' + stock + '.png')
    hook.send('High Volume detected for Ticker: ' + '**' + stock + '**' + '\n' + 'Current Price: ' + '**' + str(price) + '**' + '  |  ' + 'Percent Change: ' + '**' + str(pct) + '**', file=discord_pic)
    # pull(stock)
    # twitter(stock)
    # hook.send()
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

        
        current = df.Close.iat[-1]
        last = df.Close.iat[-2]
        pct = percentChange(last, current)
        pct = round(pct, 3)
        price = round(df['Close'].iloc[-1], 3)


        multiplier = 3
        data_mean = movingaverage(volume, 20)
        upper_limit = data_mean[-1] * multiplier
        #del df['Volume']
        df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        # print(volume.iat[-1])
        # print(upper_limit)

        if (volume.iat[-1] > upper_limit and (pct > 10.0)) or (volume.iat[-1] > upper_limit and (pct < (-10.0))):

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
            
            weekday_candlestick(stock, newAr, pct, price, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0)

            
    except Exception as e:
        print('main loop', str(e))



for n in range(length):
    word = ticker_array[n]
    graphData(word,5,13)