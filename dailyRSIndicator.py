import datetime as dt
import grequests
from bs4 import BeautifulSoup
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
import csv

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/727629742148419646/558Is3gXdN3WjQsffwh5N8X3DClL7MkC0BrHAA8WBTFIE2JtBlL1Ta1TfiR4sEFQZ-Oa")

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



def newData():
    try:
        urls = []
        for n in range(length):
            word = ticker_array[n]
            url = 'https://finance.yahoo.com/quote/' + word
            urls.append(url)
        global urlLength
        urlLength = len(urls)

        rs = (grequests.get(u) for u in urls)
        requests = grequests.map(rs, size=10)
        for response in requests:
            soup = BeautifulSoup(response.text, 'lxml')
            ticker = soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})[0].find('h1').text
            closep = soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text
            close = closep.replace(',', '')
            openp = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
            openp = openp[1].find('span').text
            openpp = openp.replace(',', '')
            high = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
            high = high[4].text
            result = [x.strip() for x in high.split(' - ')]
            highp = result[1]
            highpp = highp.replace(',', '')
            lowp = result[0]
            low = lowp.replace(',', '')
            volumep = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
            volumep = volumep[6].find('span').text
            volume = volumep.replace(',', '')
            tt = [x.strip() for x in ticker.split(' ')]
            tick = tt[0]

        
            print(tick)
            print(close)
            print(openpp)
            print(highpp)
            print(low)
            print(volume)
            now = dt.datetime.now()
            fieldnames = ["","date","close","high","low","open", "volume"]
            toFile(tick, close, now, highpp, low, openpp, volume, fieldnames)
    except IndexError as e:
        print('Error Encountered. Restarting...')
        os.execv(sys.executable, ['python'] + sys.argv)
        

def toFile(ticker, price_data, time, high, low, openn, volume, fieldnames):

    with open('dailyRSIfiles/' + ticker + '.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "": 1,
            "date": time,
            "close": price_data,
            "high": high,
            "low": low,
            "open": openn,
            "volume": volume
                    }

        csv_writer.writerow(info)

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('dailyRSIfiles/' + stock + '.csv')
        #df.index = pd.to_datetime(df.index)
        df.rename(columns={'date': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        df ['Date'] = df['Date'].str.replace('T', ' ', regex=True)
        df ['Date'] = df['Date'].str.replace('Z', '', regex=True)
        df ['Date'] = df['Date'].map(lambda x: str(x)[:-15])
        df.index.name = 'Date'
        #df = df[(df['Date'] > '2018-1-1') & (df['Date'] <= '2020-6-30')]
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(mdates.date2num)
        df = df.astype(float)
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        volume = df['Volume']

        rsi = rsiFunc(closep)

        if rsi[-1] < 30 or rsi[-1] > 70:
            
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

            fig = plt.figure(facecolor='#07000d')

            ax1 = plt.subplot2grid(
                (6, 4), (1, 0), rowspan=4, colspan=4, facecolor='#07000d')
   
            candlestick_ohlc(ax1, newAr[-SP:], colorup='green', colordown='red', width=0.6, alpha=1.0)  
           
            def standard_deviation(tf, prices):
                sd = []
                sddate = []
                x = tf
                while x <= len(prices):
                    array2consider = prices[x-tf:x]
                    standev = array2consider.std()
                    sd.append(standev)
                    sddate.append(date[x])
                    x += 1
                return sddate, sd

            def bollinger_bands(mult, tff):
                bdate = []
                topBand = []
                botBand = []
                midBand = []

                x = tff

                while x < len(date):
                    curSMA = movingaverage(closep[x-tff:x], tff)[-1]

                    d, curSD = standard_deviation(tff, closep[0:tff])
                    curSD = curSD[-1]

                    TB = curSMA + (curSD*mult)
                    BB = curSMA - (curSD*mult)
                    D = date[x]

                    bdate.append(D)
                    topBand.append(TB)
                    botBand.append(BB)
                    midBand.append(curSMA)
                    x+=1
                return bdate, topBand, botBand, midBand

            d, tb, bb, mb = bollinger_bands(2,20)
            
            
            
            
            Label1 = str(MA1)+' SMA'
            Label2 = str(MA2)+' SMA'

            #print(date)
            print(Av1)

            ax1.plot(d[-SP:], mb[-SP:], '#4ee6fd', label='20 SMA', linewidth=1 )
            ax1.plot(d[-SP:], tb [-SP:], '#32CD32', label='Upper', linewidth=3, alpha=0.5 )
            ax1.plot(d[-SP:], bb[-SP:], '#E50BD1', label='Lower', linewidth=3, alpha=0.5 )
            ax1.plot(date[-SP:], Av2[-SP:], '#FFFF00',label=Label2, linewidth=1)
            

            ax1.grid(False, color='w')
            ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.yaxis.label.set_color("w")
            ax1.spines['bottom'].set_color("#5998ff")
            ax1.spines['top'].set_color("#5998ff")
            ax1.spines['left'].set_color("#5998ff")
            ax1.spines['right'].set_color("#5998ff")
            ax1.tick_params(axis='y', colors='w')
            plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
            ax1.tick_params(axis='x', colors='w')
            plt.ylabel('Stock price')

            maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
                               fancybox=True, borderaxespad=0.)
            maLeg.get_frame().set_alpha(0.4)
            textEd = pylab.gca().get_legend().get_texts()
            pylab.setp(textEd[0:5], color='w')

            # volumeMin = 0

            ax0 = plt.subplot2grid(
                (6, 4), (0, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
           
            rsiCol = '#c1f9f7'
            posCol = '#386d13'
            negCol = '#8f2020'

            ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
            ax0.axhline(70, color=negCol)
            ax0.axhline(30, color=posCol)
            ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>= 70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
            ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<= 30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
            ax0.set_yticks([30, 70])
            ax0.yaxis.label.set_color("w")
            ax0.spines['bottom'].set_color("#5998ff")
            ax0.spines['top'].set_color("#5998ff")
            ax0.spines['left'].set_color("#5998ff")
            ax0.spines['right'].set_color("#5998ff")
            ax0.tick_params(axis='y', colors='w')
            ax0.tick_params(axis='x', colors='w')
            plt.ylabel('RSI')

            # Volume Fill Section
            volumeMin = 0
            ax1v = ax1.twinx()
            ax1v.fill_between(date[-SP:], volumeMin, volume[-SP:], facecolor='#00ffe8', alpha=.4)
            ax1v.axes.yaxis.set_ticklabels([])
            ax1v.grid(False)
            # Edit this to 3, so it's a bit larger
            ax1v.set_ylim(0, 3*volume.max())
            ax1v.spines['bottom'].set_color("#5998ff")
            ax1v.spines['top'].set_color("#5998ff")
            ax1v.spines['left'].set_color("#5998ff")
            ax1v.spines['right'].set_color("#5998ff")
            ax1v.tick_params(axis='x', colors='w')
            ax1v.tick_params(axis='y', colors='w')



            ax2 = plt.subplot2grid(
                (6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
            fillcolor = '#00ffe8'
            nslow = 26
            nfast = 12
            nema = 9
            emaslow, emafast, macd = computeMACD(closep)
            ema9 = ExpMovingAverage(macd, nema)
            ax2.plot(date[-SP:], macd[-SP:], color='#4ee6fd', lw=2)
            ax2.plot(date[-SP:], ema9[-SP:], color='#e1edf9', lw=1)
            ax2.fill_between(date[-SP:], macd[-SP:]-ema9[-SP:], 0,
                             alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

            plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
            ax2.spines['bottom'].set_color("#5998ff")
            ax2.spines['top'].set_color("#5998ff")
            ax2.spines['left'].set_color("#5998ff")
            ax2.spines['right'].set_color("#5998ff")
            ax2.tick_params(axis='x', colors='w')
            ax2.tick_params(axis='y', colors='w')
            plt.ylabel('MACD', color='w')
            ax2.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=5, prune='upper'))
            for label in ax2.xaxis.get_ticklabels():
                label.set_rotation(45)

            plt.suptitle(stock.upper(), color='w')

            plt.setp(ax0.get_xticklabels(), visible=False)
            plt.setp(ax1.get_xticklabels(), visible=False)

            

            plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
            #plt.show()
            fig.savefig('dailyRSIpics/' + stock + '.png', facecolor=fig.get_facecolor())
            discord_pic = File('dailyRSIpics/' + stock + '.png')
            hook.send("RSI ALERT: " + stock + "  Frequency: Daily", file=discord_pic)
            #os.remove('dailypics/' + stock + '.png')
            plt.close(fig)

    except Exception as e:
        # os.execv(sys.executable, ['python3'] + sys.argv)
        print(e)

newData()
for n in range(length):
    word = ticker_array[n]
    graphData(word,10,50)
