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
yf.pdr_override()

hook = Webhook("https://discordapp.com/api/webhooks/777923209931522058/hQrZoddC7QbpvWVoq8Z-eXG6tRqtddR2A-9z9HHj2RteQ8U2rb9dG2tfOfoEklBfSkzh")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TopGainers', avatar=img)

ticker_array = []
f = open('stocks.csv')
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

for n in range(length):
    word = ticker_array[n]
    Scanner(word)