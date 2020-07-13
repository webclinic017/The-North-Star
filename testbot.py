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
from collections import Counter
# from sklearn import svm, cross_validation, neighbors
# from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import matplotlib
import pylab
import os
import sys
import time
import csv

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/728005207245717635/W2mvs5RtSDPL72TmCau1vU49nfI2kJJP-yX6JzMcoiKG7-HnKPMN6R8pDTApP-V2lmqJ")


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
    
def percentChange(startPoint, currentPoint):
    return ((currentPoint-startPoint)/startPoint)*100

def exception_handler(request, exception):
    pass # suppress errors on grequests.map


def newData():
        urls = []
        for n in range(length):
            word = ticker_array[n]
            url = 'https://finance.yahoo.com/quote/' + word
            urls.append(url)
        global urlLength
        urlLength = len(urls)
        rs = (grequests.get(u) for u in urls)
        requests = grequests.map(rs)
        
        for response in requests:
            soup = BeautifulSoup(response.text, 'lxml')
            #if soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'}):
            try:
                tk = soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})[0].find('h1').text
            
                ticker = tk
                tt = [x.strip() for x in ticker.split(' ')]
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
            
                    #volume = soup.find_all('div', {'class': 'D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)'})[7].find('span').text
            
            print(tick)
            print(close)
            print(openpp)
            print(highpp)
            print(low)
            now = dt.datetime.now()
            # if now.hour > 17:
            #     quit()
            fieldnames = ["","date","close","high","low","open"]
            toFile(tick, close, now, highpp, low, openpp, fieldnames)
        
            

def toFile(ticker, price_data, time, high, low, openn, fieldnames):
    with open('testFiles/' + ticker + '.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "": 1,
            "date": time,
            "close": price_data,
            "high": high,
            "low": low,
            "open": openn
            #"prediction": 0
                    }

        csv_writer.writerow(info)

def predictData(stock, days, df):
      

    df['prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    forecast_time = int(days)

    X = np.array(df.drop(['prediction'], 1))
    Y = np.array(df['prediction'])
    X = preprocessing.scale(X)
    X_prediction = X[-forecast_time:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    #Performing the Regression on the training data
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    prediction = (clf.predict(X_prediction))

    last_row = df['Close'].iloc[-1]
    #print(last_row['Close'])
    pctChange = percentChange(float(last_row),float(prediction[3]))

    if (abs(pctChange) > 5):
        output = ("\n\nStock:" + str(stock) + "\nPrior Close:\n" + str(last_row) + "\n\nPrediction in 1 Day: " + 
        str(prediction[0]) + "\nPrediction in 4 Days: " + str(prediction[3]))
        print(output)


def graphData(stock, MA1, MA2):
    try:
        df = pd.read_csv('testFiles/' + stock + '.csv')
        df.rename(columns={'date': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
        del df['adjClose']
        del df['adjOpen']
        del df['adjLow']
        del df['adjHigh']
        del df['adjVolume']
        del df['divCash']
        del df['splitFactor']

        df ['Date'] = df['Date'].str.replace('T', ' ', regex=True)
        df ['Date'] = df['Date'].str.replace('Z', '', regex=True)
        df ['Date'] = df['Date'].map(lambda x: str(x)[:-15])
        df.index.name = 'Date'
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(mdates.date2num)
        df = df.astype(float)
        date = df['Date']
        closep = df['Close']
        highp = df['High']
        lowp = df['Low']
        openp = df['Open']
        #volume = df['Volume']
        df.drop_duplicates(subset ="Close", 
                     keep = False, inplace = True)
        df =df[df['Close'] !=0]

        predictData(stock, 4, df)
        




        # rsi = rsiFunc(closep)

        # if rsi[-1] < 30 or rsi[-1] > 70:
            
        #     x = 0
        #     y = len(date)
        #     newAr = []
        #     while x < y:
        #         appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
        #         newAr.append(appendLine)
        #         x += 1
        #     #print(newAr)
        #     Av1 = movingaverage(closep, MA1)
        #     Av2 = movingaverage(closep, MA2)

        #     SP = len(date[MA2-1:])
        #     #print(SP)
            

        #     fig = plt.figure(facecolor='#07000d')

        #     ax1 = plt.subplot2grid(
        #         (6, 4), (1, 0), rowspan=4, colspan=4, facecolor='#07000d')
        #     #print(date)
        #     candlestick_ohlc(ax1, newAr[-SP:], colorup='green', colordown='red', width=0.6, alpha=1.0)  
        #     def standard_deviation(tf, prices):
        #         sd = []
        #         sddate = []
        #         x = tf
        #         while x <= len(prices):
        #             array2consider = prices[x-tf:x]
        #             standev = array2consider.std()
        #             sd.append(standev)
        #             sddate.append(date[x])
        #             x += 1
        #         return sddate, sd

        #     def bollinger_bands(mult, tff):
        #         bdate = []
        #         topBand = []
        #         botBand = []
        #         midBand = []

        #         x = tff

        #         while x < len(date):
        #             curSMA = movingaverage(closep[x-tff:x], tff)[-1]

        #             d, curSD = standard_deviation(tff, closep[0:tff])
        #             curSD = curSD[-1]

        #             TB = curSMA + (curSD*mult)
        #             BB = curSMA - (curSD*mult)
        #             D = date[x]

        #             bdate.append(D)
        #             topBand.append(TB)
        #             botBand.append(BB)
        #             midBand.append(curSMA)
        #             x+=1
        #         return bdate, topBand, botBand, midBand

        #     d, tb, bb, mb = bollinger_bands(2,20)
            
            
            
            
        #     Label1 = str(MA1)+' SMA'
        #     Label2 = str(MA2)+' SMA'

        #     #print(date)
        #     #print(Av1)

        #     ax1.plot(d[-SP:], mb[-SP:], '#4ee6fd', label='20 SMA', linewidth=1 )
        #     ax1.plot(d[-SP:], tb [-SP:], '#32CD32', label='Upper', linewidth=3, alpha=0.5 )
        #     ax1.plot(d[-SP:], bb[-SP:], '#E50BD1', label='Lower', linewidth=3, alpha=0.5 )
        #     ax1.plot(date[-SP:], Av2[-SP:], '#FFFF00',label=Label2, linewidth=1)
            
            

        #     ax1.grid(False, color='w')
        #     ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        #     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #     ax1.yaxis.label.set_color("w")
        #     ax1.spines['bottom'].set_color("#5998ff")
        #     ax1.spines['top'].set_color("#5998ff")
        #     ax1.spines['left'].set_color("#5998ff")
        #     ax1.spines['right'].set_color("#5998ff")
        #     ax1.tick_params(axis='y', colors='w')
        #     plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        #     ax1.tick_params(axis='x', colors='w')
        #     plt.ylabel('Stock price')

        #     maLeg = plt.legend(loc=9, ncol=2, prop={'size': 7},
        #                        fancybox=True, borderaxespad=0.)
        #     maLeg.get_frame().set_alpha(0.4)
        #     textEd = pylab.gca().get_legend().get_texts()
        #     pylab.setp(textEd[0:5], color='w')

        #     # volumeMin = 0

        #     ax0 = plt.subplot2grid(
        #         (6, 4), (0, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
        #     # rsi = rsiFunc(closep)
        #     rsiCol = '#c1f9f7'
        #     posCol = '#386d13'
        #     negCol = '#8f2020'

        #     ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
        #     ax0.axhline(70, color=negCol)
        #     ax0.axhline(30, color=posCol)
        #     ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>= 70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
        #     ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<= 30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
        #     ax0.set_yticks([30, 70])
        #     ax0.yaxis.label.set_color("w")
        #     ax0.spines['bottom'].set_color("#5998ff")
        #     ax0.spines['top'].set_color("#5998ff")
        #     ax0.spines['left'].set_color("#5998ff")
        #     ax0.spines['right'].set_color("#5998ff")
        #     ax0.tick_params(axis='y', colors='w')
        #     ax0.tick_params(axis='x', colors='w')
        #     plt.ylabel('RSI')

        #     # ax1v = ax1.twinx()
        #     # ax1v.fill_between(date[-SP:], volumeMin, volume[-SP:], facecolor='#00ffe8', alpha=.4)
        #     # ax1v.axes.yaxis.set_ticklabels([])
        #     # ax1v.grid(False)
        #     # # Edit this to 3, so it's a bit larger
        #     # ax1v.set_ylim(0, 3*volume.max())
        #     # ax1v.spines['bottom'].set_color("#5998ff")
        #     # ax1v.spines['top'].set_color("#5998ff")
        #     # ax1v.spines['left'].set_color("#5998ff")
        #     # ax1v.spines['right'].set_color("#5998ff")
        #     # ax1v.tick_params(axis='x', colors='w')
        #     # ax1v.tick_params(axis='y', colors='w')
        #     ax2 = plt.subplot2grid(
        #         (6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
        #     fillcolor = '#00ffe8'
        #     nslow = 26
        #     nfast = 12
        #     nema = 9
        #     emaslow, emafast, macd = computeMACD(closep)
        #     ema9 = ExpMovingAverage(macd, nema)
        #     ax2.plot(date[-SP:], macd[-SP:], color='#4ee6fd', lw=2)
        #     ax2.plot(date[-SP:], ema9[-SP:], color='#e1edf9', lw=1)
        #     ax2.fill_between(date[-SP:], macd[-SP:]-ema9[-SP:], 0,
        #                      alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)

        #     plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
        #     ax2.spines['bottom'].set_color("#5998ff")
        #     ax2.spines['top'].set_color("#5998ff")
        #     ax2.spines['left'].set_color("#5998ff")
        #     ax2.spines['right'].set_color("#5998ff")
        #     ax2.tick_params(axis='x', colors='w')
        #     ax2.tick_params(axis='y', colors='w')
        #     plt.ylabel('MACD', color='w')
        #     ax2.yaxis.set_major_locator(
        #         mticker.MaxNLocator(nbins=5, prune='upper'))
        #     for label in ax2.xaxis.get_ticklabels():
        #         label.set_rotation(45)

        #     plt.suptitle(stock.upper(), color='w')

        #     plt.setp(ax0.get_xticklabels(), visible=False)
        #     plt.setp(ax1.get_xticklabels(), visible=False)

            
        #     print('Hit')
        #     plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
        #     #plt.show()
        #     #fig.savefig('hourRSIpics/' + stock + '.png', facecolor=fig.get_facecolor())
        #     #discord_pic = File('hourRSIpics/' + stock + '.png')
        #     #hook.send("RSI ALERT: " + stock + "  Frequency: 1 hour", file=discord_pic)
        #     plt.close(fig)

    except Exception as e:
        print('main loop', str(e))
newData()
for n in range(length):
    word = ticker_array[n]
    graphData(word,10,50)










# def process_data_for_labels(ticker):
#     hm_days = 7
#     df = pd.read_csv('hourRSIfiles/' + stock + '.csv')
#     tickers = df.columns.values.tolist()
#     df.fillna(0, inplace=True)

#     for i in range(1, hm_days+1):
#         df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

# def extract_featuresets(ticker):
#     tickers, df = process_data_for_labels(ticker)

#     df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
#                                                df['{}_1d'.format(ticker)],
#                                                df['{}_2d'.format(ticker)],
#                                                df['{}_3d'.format(ticker)],
#                                                df['{}_4d'.format(ticker)],
#                                                df['{}_5d'.format(ticker)],
#                                                df['{}_6d'.format(ticker)],
#                                                df['{}_7d'.format(ticker)] ))

#     vals = df['{}_target'.format(ticker)].values.tolist()
#     str_vals = [str(i) for i in vals]
#     print('Data spread:',Counter(str_vals))

#     df.fillna(0, inplace=True)
#     df = df.replace([np.inf, -np.inf], np.nan)
#     df.dropna(inplace=True)

#     df_vals = df[[ticker for ticker in tickers]].pct_change()
#     df_vals = df_vals.replace([np.inf, -np.inf], 0)
#     df_vals.fillna(0, inplace=True)

#     X = df_vals.values
#     y = df['{}_target'.format(ticker)].values

#     return X,y,df

# def buy_sell_hold(*args):
#     cols = [c for c in args]
#     requirement = 0.02
#     for col in cols:
#         if col > requirement:
#             return 1
#         if col < -requirement:
#             return -1
#     return 0

# def do_ml(ticker):
#     X, y, df = extract_featuresets(ticker)

#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

#     clf = VotingClassifier([('lsvc', svm.LinearSVC()),
#                             ('knn', neighbors.KNeighborsClassifier()),
#                             ('rfor', RandomForestClassifier())])

#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print('accuracy:', confidence)
#     predictions = clf.predict(X_test)
#     print('predicted class counts:', Counter(predictions))
#     print()
#     print()
#     return confidence

    # with open("sp500tickers.pickle","rb") as f:
    # tickers = pickle.load(f)

# accuracies = []
# for count,ticker in enumerate(tickers):

#     if count%10==0:
#         print(count)

#     accuracy = do_ml(ticker)
#     accuracies.append(accuracy)
#     print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))