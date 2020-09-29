from geminipy import Geminipy
import json
import pandas as pd
import numpy as np
import math
from finta import TA
import trendln
import random
from trendln import plot_support_resistance, plot_sup_res_date, plot_sup_res_learn, calc_support_resistance,METHOD_PROBHOUGH, METHOD_HOUGHPOINTS, METHOD_NCUBED, METHOD_NUMDIFF  

# The connection defaults to the Gemini sandbox.
# Add 'live=True' to use the live exchange
con = Geminipy(api_key='account-SVCu6zuYuO7LHmlgS220', secret_key='27yknygXwV2AtY5FcVdjVgwoP2aM')
    
# public request
symbols = con.symbols()
#print(symbols.content)
print(random.randint(3, 1000))
#print(dfp['type'][0])
#print(con.active_orders().content)
#print(con.past_trades().content)


btc = con.tickerCandleHistLive(symbol='btcusd', interval='1m')
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
df['s2'] = dfb['s2']
df['s2'] = df['s2'].ffill()
# (minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows) = calc_support_resistance(df['close'], #as per h for calc_support_resistance
#     method = METHOD_NUMDIFF,
# 	errpct = 0.005,
# 	hough_scale=0.01,
#     window=50,
# 	hough_prob_iter=200,
# 	sortError=False,
# 	accuracy=1)

# #print('Calculations Done. Running Backtest')
# df['MinPoint'] = df.iloc[minimaIdxs]['low']
# df['MinPoint'] = df['MinPoint'].ffill()
# df['MaxPoint'] = df.iloc[maximaIdxs]['high']
# df['MaxPoint'] = df['MaxPoint'].ffill()
#print(df)
bid_ask = con.pubticker(symbol='btcusd')
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

cash = float(dfbb['available'][1])
#print(len(dfbb['available']))
sellBalance = 0
sellcash = 0
if len(dfbb['available']) > 1:
    sellcash = float(dfbb['available'][5])
    #sellBalance = float(sellcash)
cashBuyAmount = float(cash*.95)
#WcashBuyAmount = float(cashBuyAmount + (cashBuyAmount*.01))
#sellAmount = float(dfbb['available'][5])
buypercent = float(cash/ask)
#print(buypercent)
#percent = float(math.floor(percent))
buypercent = float(buypercent*.95)
buypercent = round(buypercent, 2)

#sellpercent = float(sellcash/bid)
#print(percent)
#percent = float(math.floor(percent))
sellpercent = float(sellcash*.95)
sellpercent = round(sellpercent, 4)

# print('If Buy Request, Takes Available USD: ' + str(cash) + ' Then buys for ask of: ' + str(ask) + ' with amount of ' + str(buypercent) + ' ETH' + ' and ' + str(cashBuyAmount) + ' USD')
# print('If Sell Request, Takes Available ETH Balance of: ' + str(sellcash) + ' and The Sells for bid of: ' + str(bid) + 'with amount of ' + str(sellpercent) + 'ETH and ' + str(sellBalance) + ' USD')



#print(sellAmount)
#print(dfb['ask'][-1])

activeOrders = json.loads(con.active_orders().content)
dfa = pd.DataFrame(activeOrders)
pastTrades = json.loads(con.past_trades(symbol='ethusd').content)
dfp = pd.DataFrame(pastTrades)
print(dfp)
print(df)
df.to_csv('TESTDATA.csv')
# dfp = dfp.sort_values(by=['timestamp'])
# dfp['date'] = pd.to_datetime(dfp['timestampms'], unit='ms')
# dfp = dfp[::-1].reset_index()
#order = con.new_order(symbol='btcusd',amount='.01', price=dfb['ask'][-1],side='buy')
#print(dfp)


#BUY
#if dfp.empty:
print(ask)
print(buypercent)
print(len(activeOrders))
order = con.new_order(symbol='btcusd', amount=buypercent, price=ask,side='buy')
# if df['close'][-1] > df['MaxPoint'][-1]:

#     if len(activeOrders) == 0 and dfp['type'][0] == 'Sell' or dfp['type'][0] == 0:
#         order = con.new_order(symbol='ethusd', amount=buypercent, price=ask,side='buy', options=["immediate-or-cancel"])

# #SELL

# if df['close'][-1] < df['MinPoint'][-1] or df['close'][-1] < df['s2'][-1]:
#     if dfp['type'][0] == 'Buy':
#         order = con.new_order(symbol='ethusd', amount=sellpercent, price=bid,side='sell', options=["immediate-or-cancel"])




#print(btc.content)
# a Requests response is returned.
# So we can access the HTTP reponse code,
# the raw response content, or a json object
#print (symbols.status_code)
#print (symbols.content)
#print (symbols.json())
    
# authenticated request
#order = con.new_order(amount='1', price='200',side='buy')
    
#print (order.json())
    
#send a heartbeat
#con.heartbeat()