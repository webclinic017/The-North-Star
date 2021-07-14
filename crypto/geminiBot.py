from geminipy import Geminipy
import json
import pandas as pd
import numpy as np
import math
from finta import TA
import trendln
from trendln import plot_support_resistance, get_extrema,plot_sup_res_date, plot_sup_res_learn, calc_support_resistance,METHOD_PROBHOUGH, METHOD_HOUGHPOINTS, METHOD_NCUBED, METHOD_NUMDIFF  
import datetime as dt
import random
import time
from dhooks import Webhook, File, Embed

#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/765981512646524978/wUuYe8L16SxT0A9D-XV5uM9_NQwRDR2v7G5qDHjdxCA6uIX9ZSsymODvVxvrX5aS-gvT")
with open('eth.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='ETH Trader', avatar=img)

# The connection defaults to the Gemini sandbox.
# Add 'live=True' to use the live exchange
con = Geminipy(api_key='account-FcTghjjQaZY9aL4q4JPU', secret_key='2BJ8zzaCc7k1PTDmFbdaXmoVaG2z', live=True)
    


btc = con.tickerCandleHistLive(symbol='ethusd', interval='30m')
elevations = btc.content
data = json.loads(elevations)

df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['date'], unit='ms')
df = df[::-1].reset_index()
del df['index']


df = df.set_index('date')
df.index = pd.to_datetime(df.index)

locs = df.at_time('00:00')

dfb = TA.PIVOT_FIB(locs)

df['s2'] = dfb['s2']
df['s2'] = df['s2'].ffill()

nLowest = df.iloc[-5:-1].nsmallest(1, 'low')
df['lowest'] = nLowest['low']
df['lowest'] = df['lowest'].ffill()
print(df['lowest'][-1])
minimaIdxs, maximaIdxs = get_extrema(df['close'], accuracy=5)

#print('Calculations Done. Running Backtest')
df['MinPoint'] = df.iloc[minimaIdxs]['low']
df['MinPoint'] = df['MinPoint'].ffill()
df['MaxPoint'] = df.iloc[maximaIdxs]['high']
df['MaxPoint'] = df['MaxPoint'].ffill()
# print('If Buy Request, Takes Available USD: ' + str(cash) + ' Then buys for ask of: ' + str(ask) + ' with amount of ' + str(buypercent) + ' ETH' + ' and ' + str(cashBuyAmount) + ' USD')
# print('If Sell Request, Takes Available ETH Balance of: ' + str(sellcash) + ' and The Sells for bid of: ' + str(bid) + 'with amount of ' + str(sellpercent) + 'ETH and ' + str(sellBalance) + ' USD')


pt = json.loads(con.past_trades(symbol='ethusd').content)
pt = pd.DataFrame(pt)

activeOrders = json.loads(con.active_orders().content)
dfa = pd.DataFrame(activeOrders)



print(pt['type'])

tempID = random.randint(10, 100000)
bid_ask = con.pubticker(symbol='ethusd')
content = bid_ask.content
biddata = json.loads(content)
dfb = pd.DataFrame(biddata)
print(dfb)
activeBalance = json.loads(con.balances().content)
#print(activeBalance)
dfbb = pd.DataFrame(activeBalance)

cash = float(dfbb.loc[dfbb['currency'] == 'USD', 'available'])
sellcash = float(dfbb.loc[dfbb['currency'] == 'ETH', 'available'])


sellpercent = float(sellcash*.99)
sellpercent = round(sellpercent, 4)

book = con.book(symbol='ethusd', limit_bids=6, limit_asks=6).content
#print(book)
book = json.loads(book)
dfbook = pd.DataFrame(book)

bid = float(dfbook['bids'][0]['price'])
if float(dfbook['bids'][0]['amount']) < sellpercent:
	bid = float(dfbook['bids'][1]['price'])
if float(dfbook['bids'][0]['amount']) and float(dfbook['bids'][1]['amount']) < sellpercent:
	bid = float(dfbook['bids'][2]['price'])
if float(dfbook['bids'][0]['amount']) and float(dfbook['bids'][1]['amount']) and float(dfbook['bids'][2]['amount']) < sellpercent:
	bid = float(dfbook['bids'][3]['price'])
if float(dfbook['bids'][0]['amount']) and float(dfbook['bids'][1]['amount']) and float(dfbook['bids'][2]['amount']) and float(dfbook['bids'][3]['amount']) < sellpercent:
	bid = float(dfbook['bids'][4]['price'])
if float(dfbook['bids'][0]['amount']) and float(dfbook['bids'][1]['amount']) and float(dfbook['bids'][2]['amount']) and float(dfbook['bids'][3]['amount']) and float(dfbook['bids'][4]['amount']) < sellpercent:
	bid = float(dfbook['bids'][5]['price'])

ask = float(dfbook['asks'][0]['price'])
buypercent = float(cash/ask)
#print(buypercent)
buypercent = float(buypercent*.99)
buypercent = round(buypercent, 4)
if float(dfbook['asks'][0]['amount']) < buypercent:
	ask = float(dfbook['asks'][1]['price'])
if float(dfbook['asks'][0]['amount']) and float(dfbook['asks'][1]['amount']) < buypercent:
	ask = float(dfbook['asks'][2]['price'])
if float(dfbook['asks'][0]['amount']) and float(dfbook['asks'][1]['amount']) and float(dfbook['asks'][2]['amount']) < buypercent:
	ask = float(dfbook['asks'][3]['price'])
if float(dfbook['asks'][0]['amount']) and float(dfbook['asks'][1]['amount']) and float(dfbook['asks'][2]['amount']) and float(dfbook['asks'][3]['amount']) < buypercent:
	ask = float(dfbook['asks'][4]['price'])
if float(dfbook['asks'][0]['amount']) and float(dfbook['asks'][1]['amount']) and float(dfbook['asks'][2]['amount']) and float(dfbook['asks'][3]['amount']) and float(dfbook['asks'][4]['amount'])< buypercent:
	ask = float(dfbook['asks'][5]['price'])


s2 = round(df['s2'][-1], 2)



if df['close'][-1] > df['MaxPoint'][-1]:
	if len(activeOrders) == 0 and pt['type'][0] == 'Sell':
		order = con.new_order(symbol='ethusd', amount=buypercent, price=ask, side='buy', options=["fill-or-kill"])

		embed = Embed(
            description='Buy',
            color=0x5CDBF0,
            timestamp='now'  # sets the timestamp to current time
            )


		embed.set_author(name='Trade Executed')
		embed.add_field(name='Current Price', value= '$' + str(df['close'][-1]))
		hook.send(embed=embed)




#SELL

if df['close'][-1] < df['lowest'][-1]:
	if pt['type'][0] == 'Buy':
		order = con.new_order(symbol='ethusd', amount=sellpercent, price=bid, side='sell', options=["fill-or-kill"])

		embed = Embed(
            description='Sell',
            color=0xff0000,
            timestamp='now'  # sets the timestamp to current time
            )

		embed.set_author(name='Trade Executed')
		embed.add_field(name='Current Price', value= '$' + str(df['close'][-1]))
		hook.send(embed=embed)


df.to_csv('LIVEDATA.csv')
