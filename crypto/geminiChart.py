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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
import pylab




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
df['date'] = df['date'].dt.tz_localize('UTC')
df['date'] = df['date'].dt.tz_convert('US/Eastern')
df['date'] = df['date'].apply(lambda x: dt.datetime.replace(x, tzinfo=None))
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
print(df[df.index.duplicated()])

#print('Calculations Done. Running Backtest')
df['MinPoint'] = df.iloc[minimaIdxs]['low']
df['MinPoint'] = df['MinPoint'].ffill()
df['MaxPoint'] = df.iloc[maximaIdxs]['high']
df['MaxPoint'] = df['MaxPoint'].ffill()


pt = json.loads(con.past_trades(symbol='ethusd').content)
pt = pd.DataFrame(pt)
#pt = pt.reindex(index=pt.index[::-1])
# Date_Time = pd.to_datetime(df.NameOfColumn, unit='ms')
pt['date'] = pd.to_datetime(pt['timestampms'], unit='ms')

#Converts to EST Timezone from UTC
pt['date'] = pt['date'].dt.tz_localize('UTC')
pt['date'] = pt['date'].dt.tz_convert('US/Eastern')
pt['date'] = pt['date'].apply(lambda x: dt.datetime.replace(x, tzinfo=None))
#Rounds down to nearest 5 minutes
pt['date'] = pt['date'].apply(lambda x: dt.datetime(x.year, x.month, x.day, x.hour,5*(x.minute // 5)))
#pt['date'] = pt['date'].apply(lambda x: pd.to_datetime(x).tz_localize('US/Eastern'))
df['buy_signal'] = 0
df['sell_signal'] = 0
pt['buy_signal'] = 0
pt['sell_signal'] = 0

sell_dates = []
buy_dates = []

for i in range(len(pt)):
    if pt['type'][i] == 'Sell':
        #pt['buy_signal'][i] = df.iloc[pt[date][i]]['low']
        pt['sell_signal'][i] = 1
        sell_dates.append(pt['date'][i])
        df['sell_signal'][pt['date'][i]] = 1
    if pt['type'][i] == 'Buy':
        #pt['buy_signal'][i] = df.iloc[pt[date][i]]['low']
        pt['buy_signal'][i] = 1 
        buy_dates.append(pt['date'][i])
        df['buy_signal'][pt['date'][i]] = 1
        
# for i in range(len(buy_dates)):
#     df['buy_signal'] = df.iloc[df.index]['True']
#     df['sell_signal'] = df.iloc[sell_dates[i]]['False']

print(pt)

# df['dateTrue'] = False
# buy_dates = pd.to_datetime(buy_dates)
# buy_dates = pd.to_datetime(sell_dates)
# print(buy_dates)
date = df.index.values

# for i in range(len(buy_dates)):
buy_signals = [item for item in date if item in buy_dates]
sell_signals = [item for item in date if item in sell_dates]
print(buy_signals)


# date = pd.to_datetime(df.index, unit='ms')
# closep = df['close']
# highp = df['high']
# lowp = df['low']
# openp = df['open']



# x = 0
# y = len(df.index)
# newAr = []
# while x < y:
#     appendLine = date[x], openp[x], highp[x], lowp[x], closep[x]
#     newAr.append(appendLine)
#     x += 1


# # df['sell_signal'] = df['sell_signal'][:-30]
# # print(len(df.index))
# # print(len(df['sell_signal']))


df.to_csv('chartdata.csv')

## plotting the buy and sellsignals on graph
plt.plot(df.index, df['close'], label='Close')
ax = plt.gca()
ax.set_facecolor('black')
# candlestick_ohlc(ax, newAr)

plt.plot(df.loc[df.buy_signal == 1].index,df.close[df.buy_signal == 1],'^', markersize=8, color='g')
plt.plot(df.loc[df.sell_signal == 1].index,df.close[df.sell_signal == 1],'v', markersize=8, color='r')
#plt.plot(df.index,df.iloc[df['sell_signal' == True]['close']], color='red', markersize=10, marker="v")

## Adding labels
plt.xlabel('Date')  
plt.ylabel('Close Price')  
plt.title('Past Week of Automated Trades') 

plt.show()

# ax.plot(df.loc[df.signal == 1.0].index,
#          closep[df.signal == 1.0],
#          '^', markersize=10, color='m')



