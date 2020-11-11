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

con = Geminipy(api_key='account-FcTghjjQaZY9aL4q4JPU', secret_key='2BJ8zzaCc7k1PTDmFbdaXmoVaG2z', live=True)
    
# public request
#symbols = con.symbols()
#print(symbols.content)

#print(dfp['type'][0])
#print(con.active_orders().content)
#print(con.past_trades().content)


btc = con.tickerCandleHistLive(symbol='ethusd', interval='30m')
elevations = btc.content
data = json.loads(elevations)
#print(data)
df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
#df['date'] = df['date'].dt.to_pydatetime()
df['date'] = pd.to_datetime(df['date'], unit='ms')
df = df[::-1].reset_index()
del df['index']

#print(df)
#df.index.names = ['date']
df = df.set_index('date')
df.index = pd.to_datetime(df.index)
#print(df)
locs = df.at_time('00:00')

dfb = TA.PIVOT_FIB(locs)
#print(dfb)
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

# instantQuote = con.getQuote(side='buy', totalSpend=buypercent)
# instantQuoteS = con.getQuote(side='sell', totalSpend=sellpercent)
# #print(instantQuote.content)
# print(instantQuoteS.content)
pt = json.loads(con.past_trades(symbol='ethusd').content)
pt = pd.DataFrame(pt)
#print(pt)
#print(pt['type'][0])
#print(sellAmount)
#print(dfb['ask'][-1])
activeOrders = json.loads(con.active_orders().content)
dfa = pd.DataFrame(activeOrders)

#pasttrades = pd.read_csv('ordersBTC.csv')
#print(pasttrades['orderType'].iloc[-1])
# pastTrades = json.loads(con.past_trades().content)
# dfp = pd.DataFrame(pastTrades)

# print(dfp)
# print(df)
# print(cash)
# print(sellcash)
# print(buypercent)
# print(sellpercent)

# dfp = dfp.sort_values(by=['timestamp'])
# dfp['date'] = pd.to_datetime(dfp['timestampms'], unit='ms')
# dfp = dfp[::-1].reset_index()
#order = con.new_order(symbol='btcusd',amount='.01', price=dfb['ask'][-1],side='buy')
#print(dfp)

print(pt['type'])

#book = con.book(symbol='btcusd', limit_bids=0, limit_asks=0).content
# book = json.loads(book)
# dfbook = pd.DataFrame(book)
#print(book)
# print(dfos['is_cancelled'])
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
#print(s2)
#print(pt['type'][0])


embed = Embed(
            description='Buy',
            color=0x5CDBF0,
            timestamp='now'  # sets the timestamp to current time
            )


embed.set_author(name='Trade Executed')
embed.add_field(name='Current Price', value= '$' + str(df['close'][-1]))
hook.send(embed=embed)


embed = Embed(
            description='Sell',
            color=0xff0000,
            timestamp='now'  # sets the timestamp to current time
            )

embed.set_author(name='Trade Executed')
embed.add_field(name='Current Price', value= '$' + str(df['close'][-1]))
hook.send(embed=embed)
