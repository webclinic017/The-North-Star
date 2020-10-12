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




# The connection defaults to the Gemini sandbox.
# Add 'live=True' to use the live exchange
con = Geminipy(api_key='account-FcTghjjQaZY9aL4q4JPU', secret_key='2BJ8zzaCc7k1PTDmFbdaXmoVaG2z', live=True)
    



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

nLowest = df.iloc[-5:-1].nsmallest(1, 'low')
df['lowest'] = nLowest['low']
df['lowest'] = df['lowest'].ffill()
#print(df['lowest'][-1])
minimaIdxs, maximaIdxs = get_extrema(df['close'], accuracy=5)

#print('Calculations Done. Running Backtest')
df['MinPoint'] = df.iloc[minimaIdxs]['low']
df['MinPoint'] = df['MinPoint'].ffill()
df['MaxPoint'] = df.iloc[maximaIdxs]['high']
df['MaxPoint'] = df['MaxPoint'].ffill()


pt = json.loads(con.past_trades(symbol='ethusd').content)
pt = pd.DataFrame(pt)

# Date_Time = pd.to_datetime(df.NameOfColumn, unit='ms')
pt['date'] = pd.to_datetime(pt['timestampms'], unit='ms')

#Converts to EST Timezone from UTC
pt['date'] = pt['date'].dt.tz_localize('UTC')
pt['date'] = pt['date'].dt.tz_convert('US/Eastern')
pt['date'] = pt['date'].apply(lambda x: dt.datetime.replace(x, tzinfo=None))
#Rounds down to nearest 5 minutes
pt['date'] = pt['date'].apply(lambda x: dt.datetime(x.year, x.month, x.day, x.hour,5*(x.minute // 5)))
#pt['date'] = pt['date'].apply(lambda x: pd.to_datetime(x).tz_localize('US/Eastern'))
df['buy_signal'] = False
df['sell_signal'] = False
pt['buy_signal'] = False
pt['sell_signal'] = False

for i in range(len(pt)):
    if pt['type'][i] == 'Sell':
        #pt['buy_signal'][i] = df.iloc[pt[date][i]]['low']
        pt['sell_signal'][i] = True 
    if pt['type'][i] == 'Buy':
        #pt['buy_signal'][i] = df.iloc[pt[date][i]]['low']
        pt['buy_signal'][i] = True 

df['buy_signal'][df.index == pt['date']] = True
df['buy_signal'][df.index == pt['date']] = True

for i in range(len(df)):
    if df['index'][i] == pt['date'][i]:


print(pt['buy_signal'])
print(pt['sell_signal'])
# print(pt['date'])
# print(df)
#print(pt['type'][0])
#print(sellAmount)
#print(dfb['ask'][-1])
# activeOrders = json.loads(con.active_orders().content)
# dfa = pd.DataFrame(activeOrders)



#print(pt['type'])

# tempID = random.randint(10, 100000)
# bid_ask = con.pubticker(symbol='ethusd')
# content = bid_ask.content
# biddata = json.loads(content)
# dfb = pd.DataFrame(biddata)
# print(dfb)
# activeBalance = json.loads(con.balances().content)
# #print(activeBalance)
# dfbb = pd.DataFrame(activeBalance)

# cash = float(dfbb.loc[dfbb['currency'] == 'USD', 'available'])
# sellcash = float(dfbb.loc[dfbb['currency'] == 'ETH', 'available'])


# sellpercent = float(sellcash*.99)
# sellpercent = round(sellpercent, 4)



