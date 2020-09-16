from dhooks import Webhook, Embed, File
import robin_stocks as r
import json
import pandas as pd
import time
import datetime as dt
import csv
import matplotlib.pyplot as plt
import matplotlib
import pylab
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/733027800516263946/fc7y2ZpeMG17sYp3bWAykpb3paDcgxJhw8nXaCBUqAz9gDaJDUhT44zuMCfkdT4ypx7C")
with open('bow.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TopMovers', avatar=img)

#Robinhood Login
content = open('robinhood_info.json').read()
config = json.loads(content)
login = r.login(config['username'],config['password'])


def popularityData():
    up_movers = r.markets.get_top_movers('up', info='symbol')
    down_movers = r.markets.get_top_movers('down', info='symbol')

    for i in range(len(up_movers)):
        sector = r.stocks.get_fundamentals(up_movers[i], info='sector')
        sector=str(sector).strip("[]")
        sector=str(sector).strip("''")
        industry = r.stocks.get_fundamentals(up_movers[i], info='industry')
        industry=str(industry).strip("[]")
        industry=str(industry).strip("''")
        price = r.get_latest_price(up_movers[i])
        price=str(price).strip("[]")
        price=str(price).strip("''")
        price = round(float(price), 2)

        data = pdr.get_data_yahoo(up_movers[i], period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
        sma7 = data['Close'].rolling(7).mean()
        x = []
        for n in range(len(data)):
            x.append(n)
            
        fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
        plt.plot(x, data['Close'], color='#07000d')
        plt.plot(x, sma7, color='#5998ff')
        plt.axis('off')
        
        fig.savefig('hourPics/' + str(up_movers[i]) + '.png')
  
        embed = Embed(
            description='Upmover :arrow_up_small:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
            color=0x5CDBF0,
            timestamp='now'  # sets the timestamp to current time
            )


        image2 = 'https://i.imgur.com/f1LOr4q.png'

        embed.set_author(name=str(up_movers[i]))

        embed.add_field(name='Current Price', value= '$' + str(price))
        embed.set_footer(text='Hourly Graph of the Past Week. Blue line is 7 SMA')
        discord_pic = File('hourPics/' + str(up_movers[i]) + '.png', name= 'stock.png')
        embed.set_image('attachment://stock.png')
        hook.send(embed=embed, file = discord_pic)
        plt.close(fig)


    for i in range(len(down_movers)):
        sector = r.stocks.get_fundamentals(down_movers[i], info='sector')
        sector=str(sector).strip("[]")
        sector=str(sector).strip("''")
        industry = r.stocks.get_fundamentals(down_movers[i], info='industry')
        industry=str(industry).strip("[]")
        industry=str(industry).strip("''")
        price = r.get_latest_price(down_movers[i])
        price=str(price).strip("[]")
        price=str(price).strip("''")
        price = round(float(price), 2)

        data = pdr.get_data_yahoo(down_movers[i], period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
        sma7 = data['Close'].rolling(7).mean()
        x = []
        for n in range(len(data)):
            x.append(n)
            
        fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
        plt.plot(x, data['Close'], color='#07000d')
        plt.plot(x, sma7, color='#FF0000')
        plt.axis('off')

        
        fig.savefig('hourPics/' + str(down_movers[i]) + '.png')
        embed = Embed(
            description='Downmover :small_red_triangle_down:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
            color=0xff0000,
            timestamp='now'  # sets the timestamp to current time
            )

        image2 = 'https://i.imgur.com/f1LOr4q.png'

        embed.set_author(name=str(down_movers[i]))
        embed.add_field(name='Current Price', value='$' + str(price))
        embed.set_footer(text='Hourly Graph of the Past Week. Red line is 7 SMA')

        discord_pic = File('hourPics/' + str(down_movers[i]) + '.png', name= 'stock.png')
        embed.set_image('attachment://stock.png')

        hook.send(embed=embed, file = discord_pic)       
        plt.close(fig)


popularityData()
