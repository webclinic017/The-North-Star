from geminipy import Geminipy
import json
import pandas as pd
import numpy as np
import math
from finta import TA
import trendln
from trendln import plot_support_resistance, plot_sup_res_date, plot_sup_res_learn, calc_support_resistance,METHOD_PROBHOUGH, METHOD_HOUGHPOINTS, METHOD_NCUBED, METHOD_NUMDIFF  
import datetime as dt
import random
# The connection defaults to the Gemini sandbox.
# Add 'live=True' to use the live exchange
con = Geminipy(api_key='account-FcTghjjQaZY9aL4q4JPU', secret_key='2BJ8zzaCc7k1PTDmFbdaXmoVaG2z', live=True)
    
# public request
#symbols = con.symbols()
#print(symbols.content)

#print(dfp['type'][0])
#print(con.active_orders().content)
#print(con.past_trades().content)


btc = con.tickerCandleHistLive(symbol='ethusd', interval='5m')
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
(minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows) = calc_support_resistance(df['close'], #as per h for calc_support_resistance
    method = METHOD_NUMDIFF,
	errpct = 0.005,
	hough_scale=0.01,
    window=50,
	hough_prob_iter=200,
	sortError=False,
	accuracy=1)

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

pasttrades = pd.read_csv('orders.csv')
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


#BUY
tempID = random.randint(10, 1000)
bid_ask = con.pubticker(symbol='ethusd')
content = bid_ask.content
biddata = json.loads(content)
dfb = pd.DataFrame(biddata)

activeBalance = json.loads(con.balances().content)
#print(activeBalance)
dfbb = pd.DataFrame(activeBalance)
print(dfbb)
# print(dfbb['currency'][0])
# print(dfbb['available'][0])
# print(dfb['ask'][-1])
ask = float(dfb['ask'][-1])
bid = float(dfb['bid'][-1])
print(ask)
print(bid)

# if len(dfbb['available']) == 1:
# 	cash = float(dfbb['available'][0])
#print(len(dfbb['available']))
sellBalance = 0
sellcash = 0
if pt['type'][0] == 'Sell':
	sellcash = float(dfbb['available'][0])
	cash = float(dfbb['available'][1])
elif pt['type'][0] == 'Buy':
	cash = float(dfbb['available'][0])
	sellcash = float(dfbb['available'][1])
	
    #sellBalance = float(sellcash)
cashBuyAmount = float(cash*.98)

buypercent = float(cash/ask)
#print(buypercent)
buypercent = float(buypercent*.98)
buypercent = round(buypercent, 4)
#print(buypercent)


sellpercent = float(sellcash*.98)
sellpercent = round(sellpercent, 4)
print(buypercent)
print(sellpercent)
#print(df['close'][-1])
# print(df['MaxPoint'][-1])
# print(df['MinPoint'][-1])
s2 = round(df['s2'][-1], 2)
#print(s2)
#print(pt['type'][0])

if df['close'][-1] > df['MaxPoint'][-1]:
	if len(activeOrders) == 0 and pt['type'][0] == 'Sell':
		order = con.new_order(symbol='ethusd', amount=buypercent, price=ask, side='buy', options=["immediate-or-cancel"])
		now = dt.datetime.now()
		typetoAppend = {'timestamp': [now], 
				'orderType': ['buy']}
		typetoAppend = pd.DataFrame(typetoAppend)
		#print(typetoAppend)
		pasttrades = pasttrades.append(typetoAppend)
		pasttrades.reset_index(drop=True)
		#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		pasttrades.to_csv('orders.csv', index=False)
		# orderStatus = con.order_status(order_id=tempID).content
		# orderStatus = json.loads(orderStatus)
		# dfos = pd.DataFrame(orderStatus)

		# if dfos['is_cancelled'].iloc[-1] == 'True':
		# 	bidask = con.pubticker(symbol='ethusd')
		# 	content = bidask.content
		# 	biddata = json.loads(content)
		# 	dfb = pd.DataFrame(biddata)
		# 	ask = float(dfb['ask'][-1])
		# 	order = con.new_order(symbol='ethusd', amount=buypercent, price=ask, side='buy', options=["immediate-or-cancel"])
		# 	now = dt.datetime.now()
		# 	typetoAppend = {'timestamp': [now], 
		# 			'orderType': ['buy']}
		# 	typetoAppend = pd.DataFrame(typetoAppend)
		# 	#print(typetoAppend)
		# 	pasttrades = pasttrades.append(typetoAppend)
		# 	pasttrades.reset_index(drop=True)
		# 	#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		# 	pasttrades.to_csv('orders.csv', index=False)
		# else:
		# 	now = dt.datetime.now()
		# 	typetoAppend = {'timestamp': [now], 
		# 			'orderType': ['buy']}
		# 	typetoAppend = pd.DataFrame(typetoAppend)
		# 	#print(typetoAppend)
		# 	pasttrades = pasttrades.append(typetoAppend)
		# 	pasttrades.reset_index(drop=True)
		# 	#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		# 	pasttrades.to_csv('orders.csv', index=False)


#SELL

if df['close'][-1] < df['MinPoint'][-1] or df['close'][-1] < s2:
	if pt['type'][0] == 'Buy':
		order = con.new_order(symbol='ethusd', amount=sellpercent, price=bid, side='sell', options=["immediate-or-cancel"])
		now = dt.datetime.now()
		typetoAppend = {'timestamp': [now], 
				'orderType': ['sell']}
		typetoAppend = pd.DataFrame(typetoAppend)
		#print(typetoAppend)
		pasttrades = pasttrades.append(typetoAppend)
		pasttrades.reset_index(drop=True)
		#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		pasttrades.to_csv('orders.csv', index=False)
		# orderStatus = con.order_status(order_id=tempID).content
		# orderStatus = json.loads(orderStatus)
		# dfos = pd.DataFrame(orderStatus)

		# if dfos['is_cancelled'].iloc[-1] == 'True':
		# 	bidask = con.pubticker(symbol='ethusd')
		# 	content = bidask.content
		# 	biddata = json.loads(content)
		# 	dfb = pd.DataFrame(biddata)
		# 	ask = float(dfb['ask'][-1])
		# 	order = con.new_order(symbol='ethusd', amount=sellpercent, price=bid, side='sell', options=["immediate-or-cancel"])
		# 	now = dt.datetime.now()
		# 	typetoAppend = {'timestamp': [now], 
		# 			'orderType': ['buy']}
		# 	typetoAppend = pd.DataFrame(typetoAppend)
		# 	#print(typetoAppend)
		# 	pasttrades = pasttrades.append(typetoAppend)
		# 	pasttrades.reset_index(drop=True)
		# 	#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		# 	pasttrades.to_csv('orders.csv', index=False)
		# else:
		# 	now = dt.datetime.now()
		# 	typetoAppend = {'timestamp': [now], 
		# 			'orderType': ['buy']}
		# 	typetoAppend = pd.DataFrame(typetoAppend)
		# 	#print(typetoAppend)
		# 	pasttrades = pasttrades.append(typetoAppend)
		# 	pasttrades.reset_index(drop=True)
		# 	#pasttrades = pd.concat([pasttrades, typetoAppend], ignore_index=True, sort=True)
		# 	pasttrades.to_csv('orders.csv', index=False)



df.to_csv('LIVEDATA.csv')
