from dhooks import Webhook
import robin_stocks as r
import json
import pandas as pd
import time
import datetime as dt
import csv



#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/733027800516263946/fc7y2ZpeMG17sYp3bWAykpb3paDcgxJhw8nXaCBUqAz9gDaJDUhT44zuMCfkdT4ypx7C")
ratinghook = Webhook("https://discordapp.com/api/webhooks/740012150369681498/Iogmvc03jOR90iMthSMJqAljxgvuWvHkuXU9fHpgQqQRTyrC-xOXwYbyYPN6sgDNkYq9")
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
    hook.send("UPMOVERS: \n")
    for i in range(len(up_movers)):
        pop = r.stocks.get_popularity(up_movers[i], info='num_open_positions')
        price = r.get_latest_price(up_movers[i])
        hook.send(str(up_movers[i]) + '  |  ' + "Popularity: "+ str(pop) + '  |  ' + 'Current Price: ' + str(price))
        #toFile(up_movers[i], pop)

        
    hook.send("DOWNMOVERS: \n")
    for i in range(len(down_movers)):
        pop2 = r.stocks.get_popularity(down_movers[i], info='num_open_positions')
        price2 = r.get_latest_price(down_movers[i])
        hook.send(str(down_movers[i]) + '  |  ' + "Popularity: "+ str(pop2) + '  |  ' + 'Current Price: ' + str(price2))
        #toFile(down_movers[i], pop2)


    ratinghook.send("UPMOVERS: \n")
    for i in range(len(up_movers)):
        ratings = r.get_ratings(up_movers[i], info='summary')
        rating_desc = r.get_ratings(up_movers[i], info='ratings')
        df = pd.DataFrame.from_dict(rating_desc) 
        df ['published_at'] = df['published_at'].str.replace('T', ' ', regex=True)
        df ['published_at'] = df['published_at'].str.replace('Z', '', regex=True)
        date = df['published_at']
        text = df['text']
        typee = df['type']
        for n in range(len(date)):
            ratinghook.send('Stock: ' + up_movers[i] + '\n' + date[n] + '\n'
             + str(text[n]) + '\n' + '-' + '\n' + 'Rating: ' + str(typee[n]) + '\n' + '-')
        ratinghook.send('Ratings Ratio: ' + str(ratings) + '\n' + '----------' + '\n' + '----------')

    ratinghook.send("DOWNMOVERS: \n")
    for i in range(len(down_movers)):
        ratings = r.get_ratings(down_movers[i], info='summary')
        rating_desc = r.get_ratings(down_movers[i], info='ratings')
        df = pd.DataFrame.from_dict(rating_desc) 
        df ['published_at'] = df['published_at'].str.replace('T', ' ', regex=True)
        df ['published_at'] = df['published_at'].str.replace('Z', '', regex=True)
        date = df['published_at']
        text = df['text']
        typee = df['type']
        for n in range(len(date)):
            ratinghook.send('Stock: ' + down_movers[i] + '\n' + date[n] + '\n'
             + str(text[n]) + '\n' + '-' + '\n' + 'Rating: ' + str(typee[n]) + '\n' + '-')
        ratinghook.send('Ratings Ratio: ' + str(ratings) + '\n' + '----------' + '\n' + '----------')
   
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