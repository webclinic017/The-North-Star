from geminipy import Geminipy
import json
import pandas as pd
import numpy as np
from finta import TA
import trendln
from trendln import plot_support_resistance, plot_sup_res_date, plot_sup_res_learn, calc_support_resistance,METHOD_PROBHOUGH, METHOD_HOUGHPOINTS, METHOD_NCUBED, METHOD_NUMDIFF  

# The connection defaults to the Gemini sandbox.
# Add 'live=True' to use the live exchange
con = Geminipy(api_key='account-SVCu6zuYuO7LHmlgS220', secret_key='27yknygXwV2AtY5FcVdjVgwoP2aM')
    
# public request
symbols = con.symbols()


#print(dfp['type'][0])
#print(con.active_orders().content)
#print(con.past_trades().content)


btc = con.tickerCandleHist(symbol='btcusd', interval='5m')
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
(minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows) = calc_support_resistance(df['close'], #as per h for calc_support_resistance
    method = METHOD_NUMDIFF,
	errpct = 0.005,
	hough_scale=0.01,
    window=50,
	hough_prob_iter=200,
	sortError=False,
	accuracy=1)

print('Calculations Done. Running Backtest')
df['MinPoint'] = df.iloc[minimaIdxs]['low']
df['MinPoint'] = df['MinPoint'].ffill()
df['MaxPoint'] = df.iloc[maximaIdxs]['high']
df['MaxPoint'] = df['MaxPoint'].ffill()

bid_ask = con.pubticker(symbol='btcusd')
content = bid_ask.content
biddata = json.loads(content)
dfb = pd.DataFrame(biddata)
balance = con.balances().content
#print(balance)
#print(dfb['ask'][-1])
activeOrders = json.loads(con.active_orders().content)
dfa = pd.DataFrame(activeOrders)
pastTrades = json.loads(con.past_trades().content)
dfp = pd.DataFrame(pastTrades)

#order = con.new_order(symbol='btcusd',amount='.01', price=dfb['ask'][-1],side='buy')
print(dfp['type'][0])

#print(df)
# if len(activeOrders) == 0 and dfp['type'][0] == 'Sell':
#         order = con.new_order(symbol='btcusd', amount='.01', price=dfb['ask'][-1],side='buy',options=["immediate-or-cancel"])

# if dfp['type'][0] == 'Buy':
#     order = con.new_order(symbol='btcusd', amount='.01', price=dfb['bid'][-1],side='sell',options=["immediate-or-cancel"])
#BUY
if df['close'][-1] > df['MaxPoint'][-1]:

    if len(activeOrders) == 0 and dfp['type'][0] == 'Sell':
        order = con.new_order(symbol='btcusd', amount='.01', price=dfb['ask'][-1],side='buy', options=["immediate-or-cancel"])

#SELL

if df['close'][-1] < df['MinPoint'][-1] or df['close'][-1] < df['s2'][-1]:
    if dfp['type'][0] == 'Buy':
        order = con.new_order(symbol='btcusd', amount='.01', price=dfb['bid'][-1],side='sell', options=["immediate-or-cancel"])




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