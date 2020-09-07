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


# now = dt.datetime.now()
# if now.hour > 16:
#     quit()
#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/728684390141526027/wSrLkpO9AlZ-Ps2r7-bKjfWCkw6AINjjE5c8HaDCDnQhAV7IWyKdl16UbghVZZW2g-XR")
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
    
def newData():
    urls = []
    for n in range(length):
        word = ticker_array[n]
        url = 'https://finance.yahoo.com/quote/' + word
        urls.append(url)
    global urlLength
    urlLength = len(urls)
        #print(urls)
    rs = (grequests.get(u) for u in urls)
    requests = grequests.map(rs, size=20)
    for response in requests:
        soup = BeautifulSoup(response.text, 'lxml')
        #if soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'}):
        try:
            tk = soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})[0].find('h1').text
            tt = [x.strip(')') for x in tk.split('(') if ')' in x]
            tick = tt[0]
        except IndexError:
            tick = 'ERROR'
            #print('Error Encountered. Restarting...')
        # os.execv(sys.executable, ['python'] + sys.argv)
        #if soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'}):
        try:
            closep = soup.find_all('div', {'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text
            close = closep.replace(',', '')
        except IndexError:
            close = 0
        
        try:
            openp = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})[1].find('span').text
            openpp = openp.replace(',', '')
        except IndexError:
            openpp = 0
        try:
            h = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
        
            high = h[4].text
            result = [x.strip() for x in high.split(' - ')]
            highp = result[1]
            highpp = highp.replace(',', '')
            lowp = result[0]
            low = lowp.replace(',', '')
        except IndexError:
            highpp = 0
            low = 0
     
                
        print(tick)
        # print(close)
        # print(openpp)
        # print(highpp)
        # print(low)
        now = dt.datetime.now()
        now = now.date()
        fieldnames = ["Date","Close","High","Low","Open","Adj Close","Volume"]
        toFile(tick, close, now, highpp, low, openpp, fieldnames)


def toFile(ticker, close, time, high, low, openn, fieldnames):
    with open('hourRSIfiles/' + ticker + '.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "Date": time,
            "Open": openn,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": 0,
            "Volume": 0
                    }

        csv_writer.writerow(info)

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
        (6, 4), (1, 0), rowspan=4, colspan=4, facecolor='#07000d')
    candlestick_ohlc(ax, ohlc_data_arr2, **kwargs)
    rsi = rsiFunc(closep)

    # Format x axis
    # def standard_deviation(tf, prices):
    #     sd = []
    #     sddate = []
    #     x = tf
    #     while x <= len(prices):
    #         array2consider = prices[x-tf:x]
    #         standev = array2consider.std()
    #         sd.append(standev)
    #         sddate.append(ndays[x])
    #         x += 1
    #     return sddate, sd

    # def bollinger_bands(mult, tff):
    #     bdate = []
    #     topBand = []
    #     botBand = []
    #     midBand = []

    #     x = tff

    #     while x < len(dates):
    #         curSMA = movingaverage(closep[x-tff:x], tff)[-1]

    #         d, curSD = standard_deviation(tff, closep[0:tff])
    #         curSD = curSD[-1]

    #         TB = curSMA + (curSD*mult)
    #         BB = curSMA - (curSD*mult)
    #         D = ndays[x]

    #         bdate.append(D)
    #         topBand.append(TB)
    #         botBand.append(BB)
    #         midBand.append(curSMA)
    #         x+=1
    #     return bdate, topBand, botBand, midBand

    # d, tb, bb, mb = bollinger_bands(2,20)
    
    
    
    
    # Label1 = '10 SMA'
    # Label2 = '50 SMA'

    # #print(date)
    # #print(Av1)

    # ax.plot(d, mb, '#4ee6fd', label='20 SMA', linewidth=1 )
    # ax.plot(d, tb , '#32CD32', label='Upper', linewidth=3, alpha=0.5 )
    # ax.plot(d, bb, '#E50BD1', label='Lower', linewidth=3, alpha=0.5 )
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
    day_labels = count / 9
    day_labels = int(round(day_labels))
    
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

    volumeMin = 0

    ax0 = plt.subplot2grid(
        (6, 4), (0, 0), sharex=ax, rowspan=1, colspan=4, facecolor='#07000d')
    # rsi = rsiFunc(closep)
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'

    ax0.plot(ndays, rsi, rsiCol, linewidth=1.5)
    ax0.axhline(70, color=negCol)
    ax0.axhline(30, color=posCol)
    ax0.fill_between(ndays, rsi, 70, where=(rsi>= 70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(ndays, rsi, 30, where=(rsi<= 30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([30, 70])
    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')

    ax1v = ax.twinx()
    ax1v.fill_between(ndays, volumeMin, volume, facecolor='#00ffe8', alpha=.4)
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
        (6, 4), (5, 0), sharex=ax, rowspan=1, colspan=4, facecolor='#07000d')
    fillcolor = '#00ffe8'
    nslow = 26
    nfast = 12
    nema = 9
    emaslow, emafast, macd = computeMACD(closep)
    ema9 = ExpMovingAverage(macd, nema)
    ax2.plot(ndays, macd, color='#4ee6fd', lw=2)
    ax2.plot(ndays, ema9, color='#e1edf9', lw=1)
    ax2.fill_between(ndays, macd-ema9, 0,
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
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid(which='major', axis='y', linestyle='-', alpha=.2)
    
    #print('Hit' + stock)
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    #plt.show()
    
    
    fig.savefig('dailyPics/' + stock + '.png', facecolor=fig.get_facecolor())
    discord_pic = File('dailyPics/' + stock + '.png')
    hook.send("MACD ALERT: " + stock + "  Frequency: Daily", file=discord_pic)
    plt.close(fig)
    
    


    
   
    
    #plt.show()

def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('dailyfilesDump/' + stock + '.csv')
        #df = df.reset_index()
        #df.index = pd.to_datetime(df.index)
        #print(df)

        df.index.name = 'Date'
        # #df = df[(df['Date'] > '2018-1-1') & (df['Date'] <= '2020-8-20')]
        df['Date'] = pd.to_datetime(df['Date'])
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
        volume = df['Volume']
        #df.drop_duplicates(subset ="Close", keep = False, inplace = True)
        # df =df[df['Close'] !=0]

        rsi = rsiFunc(closep)

        nslow = 26
        nfast = 12
        nema = 9
        emaslow, emafast, macd = computeMACD(closep)
        ema9 = ExpMovingAverage(macd, nema)
        SP = len(date[MA2-1:])
        maclength = len(macd)
        macdd = macd.tolist()
        ema99 = ema9.tolist()
        if (macdd[-1] - ema99[-1]) < .005 and (macdd[-1] - ema99[-1]) > (-.005):
            
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
            

           
            weekday_candlestick(stock, newAr, closep, openp, volume, Av1, Av2, date, SP, df, fmt='%b %d', freq=3, width=0.5, colorup='green', colordown='red', alpha=1.0)
    
            #candlestick_ohlc(ax1, newAr[-SP:], colorup='green', colordown='red', width=0.6, alpha=1.0)  
            
    except Exception as e:
        print('main loop', str(e))


#newData()
#init()
for n in range(length):
    word = ticker_array[n]
    graphData(word,10,50)