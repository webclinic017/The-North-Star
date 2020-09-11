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
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)

# sp500hook = Webhook("https://discordapp.com/api/webhooks/745670276636868778/-ZrlutAHLk1Jyd_rMKs4x0wAtcmB-O9LWr2dvGP55wa3qN1Lug96NosusF9xrac7Nrdx")
# ratinghook = Webhook("https://discordapp.com/api/webhooks/740012150369681498/Iogmvc03jOR90iMthSMJqAljxgvuWvHkuXU9fHpgQqQRTyrC-xOXwYbyYPN6sgDNkYq9")
#Robinhood Login
content = open('robinhood_info.json').read()
config = json.loads(content)
login = r.login(config['username'],config['password'])

def percentChang(startPoint, currentPoint):
    return ((currentPoint-startPoint)/startPoint)*100


def toFile(ticker, popularity):
    now = dt.datetime.now()
    date = now.replace(microsecond=0)
    fieldnames = ["date", "ticker", "popularity"]
    with open('topMoverfiles/' + ticker + '.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            csv_writer.writeheader()
            doPercentChange = False
        else:
            doPercentChange = True
            



        info = {
            "date": date,
            "ticker": ticker,
            "popularity": popularity
                    }

        csv_writer.writerow(info)

    if  doPercentChange == True:
        df = pd.read_csv('topMoverfiles/' + ticker + '.csv')
        df.index.name = 'Date'
        pop = df["popularity"]
        lst = list(pop)
        length = len(lst)
        start = lst[length-2]
        curr = lst[length-1]
        # df["prevPop"] = df["popularity"].shift()
        # prevPop = df["prevPop"]
        pc = percentChang(start, curr)
        hook.send("%" + " Pop Change= " + str(round(pc, 3)) + "%" + '\n' + '---------------')
        doPercentChange = False



def popularityData():
        
    # for i in range(length):
    #     print(ticker_array[i])
    #     print(r.stocks.get_popularity(ticker_array[i], info='num_open_positions'))
    #     print(r.stocks.get_fundamentals(ticker_array[i], info='float'))
    up_movers = r.markets.get_top_movers('up', info='symbol')
    down_movers = r.markets.get_top_movers('down', info='symbol')
    # sp500up_movers = r.markets.get_top_movers_sp500('up', info='symbol')
    # sp500down_movers = r.markets.get_top_movers_sp500('down', info='symbol')
    # r.export.export_completed_option_orders('/Users/mecia@moravian.edu/tingbot', file_name=None)
    # df = pd.read_csv('/Users/mecia@moravian.edu/tingbot/option_orders_Aug-21-2020.csv')
    # del df['processed_quantity']
    # print(df.to_string())



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
        #x = np.array(x)
        for n in range(len(data)):
            x.append(n)
            
        fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
        plt.plot(x, data['Close'], color='#07000d')
        plt.plot(x, sma7, color='#5998ff')
        plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # frame1 = plt.gca()
        # frame1.axes.xaxis.set_ticklabels([])
        
        fig.savefig('hourPics/' + str(up_movers[i]) + '.png')
        #hook.send(str(up_movers[i]) + '  |  ' + 'Current Price: ' + str(price))
        embed = Embed(
            description='Upmover :arrow_up_small:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
            color=0x5CDBF0,
            timestamp='now'  # sets the timestamp to current time
            )

        # image1 = 'https://i.imgur.com/rdm3W9t.png'
        image2 = 'https://i.imgur.com/f1LOr4q.png'

        embed.set_author(name=str(up_movers[i]))
        #embed.set_author(name='Upmover') #, icon_url=image1
        embed.add_field(name='Current Price', value= '$' + str(price))
        embed.set_footer(text='Hourly Graph of the Past Week. Blue line is 7 SMA')

        #embed.set_thumbnail(image1)
        discord_pic = File('hourPics/' + str(up_movers[i]) + '.png', name= 'stock.png')
        embed.set_image('attachment://stock.png')

        hook.send(embed=embed, file = discord_pic)
        #hook.send('Hourly Graph of the Past Week. Blue line is 10 SMA', file=discord_pic)
        
        plt.close(fig)
        
        #toFile(up_movers[i], pop)

        

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
        #x = np.array(x)
        for n in range(len(data)):
            x.append(n)
            
        fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
        plt.plot(x, data['Close'], color='#07000d')
        plt.plot(x, sma7, color='#FF0000')
        plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # frame1 = plt.gca()
        # frame1.axes.xaxis.set_ticklabels([])
        
        fig.savefig('hourPics/' + str(down_movers[i]) + '.png')
        #hook.send(str(up_movers[i]) + '  |  ' + 'Current Price: ' + str(price))
        embed = Embed(
            description='Downmover :small_red_triangle_down:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
            color=0xff0000,
            timestamp='now'  # sets the timestamp to current time
            )

        # image1 = 'https://i.imgur.com/rdm3W9t.png'
        image2 = 'https://i.imgur.com/f1LOr4q.png'

        embed.set_author(name=str(down_movers[i]))
        #embed.set_author(name='Downmover') #, icon_url=image1
        embed.add_field(name='Current Price', value='$' + str(price))
        embed.set_footer(text='Hourly Graph of the Past Week. Red line is 7 SMA')

        #embed.set_thumbnail(image1)
        discord_pic = File('hourPics/' + str(down_movers[i]) + '.png', name= 'stock.png')
        embed.set_image('attachment://stock.png')

        hook.send(embed=embed, file = discord_pic)
        #hook.send('Hourly Graph of the Past Week. Blue line is 10 SMA', file=discord_pic)
        
        plt.close(fig)

    # # for i in range(5):
    #     sector = r.stocks.get_fundamentals(sp500up_movers[i], info='sector')
    #     sector=str(sector).strip("[]")
    #     sector=str(sector).strip("''")
    #     industry = r.stocks.get_fundamentals(sp500up_movers[i], info='industry')
    #     industry=str(industry).strip("[]")
    #     industry=str(industry).strip("''")
    #     price = r.get_latest_price(sp500up_movers[i])
    #     price=str(price).strip("[]")
    #     price=str(price).strip("''")
    #     price = round(float(price), 2)

    #     data = pdr.get_data_yahoo(sp500up_movers[i], period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
    #     sma7 = data['Close'].rolling(7).mean()
    #     x = []
    #     #x = np.array(x)
    #     for n in range(len(data)):
    #         x.append(n)
            
    #     fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
    #     plt.plot(x, data['Close'], color='#07000d')
    #     plt.plot(x, sma7, color='#5998ff')
    #     plt.axis('off')
    #     # fig.axes.get_xaxis().set_visible(False)
    #     # fig.axes.get_yaxis().set_visible(False)
    #     # frame1 = plt.gca()
    #     # frame1.axes.xaxis.set_ticklabels([])
        
    #     fig.savefig('hourRSIpics/' + str(sp500up_movers[i]) + '.png')
    #     #hook.send(str(up_movers[i]) + '  |  ' + 'Current Price: ' + str(price))
    #     embed = Embed(
    #         description='Downmover :small_red_triangle_down:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
    #         color=0x5CDBF0,
    #         timestamp='now'  # sets the timestamp to current time
    #         )

    #     # image1 = 'https://i.imgur.com/rdm3W9t.png'
    #     image2 = 'https://i.imgur.com/f1LOr4q.png'

    #     embed.set_author(name=str(sp500up_movers[i]))
    #     #embed.set_author(name='Downmover') #, icon_url=image1
    #     embed.add_field(name='Current Price', value='$' + str(price))
    #     embed.set_footer(text='Hourly Graph of the Past Week. Blue line is 7 SMA')

    #     #embed.set_thumbnail(image1)
    #     discord_pic = File('hourRSIpics/' + str(sp500up_movers[i]) + '.png', name= 'stock.png')
    #     embed.set_image('attachment://stock.png')

    #     sp500hook.send(embed=embed, file = discord_pic)
    #     #hook.send('Hourly Graph of the Past Week. Blue line is 10 SMA', file=discord_pic)
        
    #     plt.close(fig)

    # for i in range(5):
    #     sector = r.stocks.get_fundamentals(sp500down_movers[i], info='sector')
    #     sector=str(sector).strip("[]")
    #     sector=str(sector).strip("''")
    #     industry = r.stocks.get_fundamentals(sp500down_movers[i], info='industry')
    #     industry=str(industry).strip("[]")
    #     industry=str(industry).strip("''")
    #     price = r.get_latest_price(sp500down_movers[i])
    #     price=str(price).strip("[]")
    #     price=str(price).strip("''")
    #     price = round(float(price), 2)

    #     data = pdr.get_data_yahoo(sp500down_movers[i], period = "5d", interval = "1h", retry=20, status_forcelist=[404, 400, 429, 500, 502, 503, 504], prepost = True)
    #     sma7 = data['Close'].rolling(7).mean()
    #     x = []
    #     #x = np.array(x)
    #     for n in range(len(data)):
    #         x.append(n)
            
    #     fig = plt.figure(figsize=(3,1), facecolor='#FFFF')
    #     plt.plot(x, data['Close'], color='#07000d')
    #     plt.plot(x, sma7, color='#5998ff')
    #     plt.axis('off')
    #     # fig.axes.get_xaxis().set_visible(False)
    #     # fig.axes.get_yaxis().set_visible(False)
    #     # frame1 = plt.gca()
    #     # frame1.axes.xaxis.set_ticklabels([])
        
    #     fig.savefig('hourRSIpics/' + str(sp500down_movers[i]) + '.png')
    #     #hook.send(str(up_movers[i]) + '  |  ' + 'Current Price: ' + str(price))
    #     embed = Embed(
    #         description='Downmover :small_red_triangle_down:' + '\n' + 'Sector: ' + str(sector) + '\n' + 'Industry: ' + str(industry),
    #         color=0xff0000,
    #         timestamp='now'  # sets the timestamp to current time
    #         )

    #     # image1 = 'https://i.imgur.com/rdm3W9t.png'
    #     image2 = 'https://i.imgur.com/f1LOr4q.png'

    #     embed.set_author(name=str(sp500down_movers[i]))
    #     #embed.set_author(name='Downmover') #, icon_url=image1
    #     embed.add_field(name='Current Price', value='$' + str(price))
    #     embed.set_footer(text='Hourly Graph of the Past Week. Blue line is 7 SMA')

    #     #embed.set_thumbnail(image1)
    #     discord_pic = File('hourRSIpics/' + str(sp500down_movers[i]) + '.png', name= 'stock.png')
    #     embed.set_image('attachment://stock.png')

    #     sp500hook.send(embed=embed, file = discord_pic)
    #     #hook.send('Hourly Graph of the Past Week. Blue line is 10 SMA', file=discord_pic)
        
    #     plt.close(fig)


    # ratinghook.send("UPMOVERS: \n")
    # for i in range(len(up_movers)):
    #     ratings = r.get_ratings(up_movers[i], info='summary')
    #     rating_desc = r.get_ratings(up_movers[i], info='ratings')
    #     df = pd.DataFrame.from_dict(rating_desc) 
    #     df ['published_at'] = df['published_at'].str.replace('T', ' ', regex=True)
    #     df ['published_at'] = df['published_at'].str.replace('Z', '', regex=True)
    #     date = df['published_at']
    #     text = df['text']
    #     typee = df['type']
    #     for n in range(len(date)):
    #         ratinghook.send('Stock: ' + up_movers[i] + '\n' + date[n] + '\n'
    #          + str(text[n]) + '\n' + '-' + '\n' + 'Rating: ' + str(typee[n]) + '\n' + '-')
    #     ratinghook.send('Ratings Ratio: ' + str(ratings) + '\n' + '----------' + '\n' + '----------')

    # ratinghook.send("DOWNMOVERS: \n")
    # for i in range(len(down_movers)):
    #     ratings = r.get_ratings(down_movers[i], info='summary')
    #     rating_desc = r.get_ratings(down_movers[i], info='ratings')
    #     df = pd.DataFrame.from_dict(rating_desc) 
    #     df ['published_at'] = df['published_at'].str.replace('T', ' ', regex=True)
    #     df ['published_at'] = df['published_at'].str.replace('Z', '', regex=True)
    #     date = df['published_at']
    #     text = df['text']
    #     typee = df['type']
    #     for n in range(len(date)):
    #         ratinghook.send('Stock: ' + down_movers[i] + '\n' + date[n] + '\n'
    #          + str(text[n]) + '\n' + '-' + '\n' + 'Rating: ' + str(typee[n]) + '\n' + '-')
    #     ratinghook.send('Ratings Ratio: ' + str(ratings) + '\n' + '----------' + '\n' + '----------')
   
#profile = r.profiles.load_portfolio_profile()
# prof = r.account.get_all_positions(info='average_buy_price')
# profile = r.account.get_all_positions()
# # df = pd.DataFrame.from_dict(profile)
# print(prof)
# print(profile)


popularityData()

# timeLoop = True

# Sec = 0
# Min = 0
# Interval = 0
# # Begin Process
# while timeLoop:
#     Sec += 1
#     print(str(Min) + " Mins " + str(Sec) + " Sec ")
#     time.sleep(1)
#     if Sec == 60:
#         Sec = 0
#         Min += 1
#         if Min == 30:
#             Interval += 1
#             Min = 0
#             popularityData()