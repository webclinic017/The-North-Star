# Load gemini
from gemini  import data, engine, helpers
# Global Imports
import pandas as pd
import numpy as np
from finta import TA
import trendln
from trendln import plot_support_resistance, get_extrema, plot_sup_res_date, plot_sup_res_learn, calc_support_resistance,METHOD_NAIVECONSEC , METHOD_HOUGHPOINTS, METHOD_NCUBED, METHOD_NUMDIFF, METHOD_NSQUREDLOGN

def movingaverage(values, window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas  # as a numpy array

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

def Vortex(df, n):  
    i = 0  
    TR = [0]  
    #print(df.iloc(i, 'high'))
    while i < len(df) - 1:  
        Range = max(df['high'][i+1], df['close'][i]) - min(df['low'][i+1], df['close'][i])
        TR.append(Range)  
        i = i + 1  
    
    i = 0  
    VM = [0]  
    while i < len(df) - 1:  
        Range = abs(df['high'][i+1] - df['low'][i]) - abs(df['low'][i+1] - df['high'][i])  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum())  
    #df = df.join(VI) 
    df['VI'] = VI.values
    window = 25
    df['signal'] = 0.0
    df['signal'][window:] = np.where(df['VI'][window:] > 0, 1.0, 0.0)
    df['positions'] = df['signal'].diff()
    # print('positions')
    # print(df['positions'][49])
    #print(df['VI'])
    return df
    
def TR(o,h,l,c,pc):    
    x = h-l
    y = abs(h-pc)
    z = abs(l-pc)
    
    if y <= x >= z:
        TR = x
    elif x <= y >= z:
        TR = y
    elif x <= z >= y:
        TR = z
    else:
        TR = 0
        
    return TR

def vol(returns):
    # Return the standard deviation of returns
    return numpy.std(returns)

def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)

# Higher timeframes (>= daily)
#df = data.get_htf_candles("BTC_USD", "Bitfinex", "3-DAY", "2019-01-12 00:00:00", "2019-02-01 00:00:00")

# Lower timeframes (< daily)
df = data.get_ltf_candles("USDT_BTC", "15-MIN", "2020-09-15 00:00:00", "2020-10-03 11:00:00")
df = df.reset_index()
df = df.set_index('date')
df.index = pd.to_datetime(df.index)
locs = df.at_time('00:00')
# locs2 = df.at_time('02:00')
# locs3 = df.at_time('04:00')
# locs4 = df.at_time('06:00')
# locs5 = df.at_time('08:00')
# locs6 = df.at_time('10:00')
# locs7 = df.at_time('12:00')
# locs8 = df.at_time('14:00')
# locs9 = df.at_time('16:00')
# locs10 = df.at_time('18:00')
# locs11 = df.at_time('20:00')
# locs12 = df.at_time('22:00')
# locs = locs.append(locs2)
# locs = locs.append(locs3)
# locs = locs.append(locs4)
# locs = locs.append(locs5)
# locs = locs.append(locs6)
# locs = locs.append(locs7)
# locs = locs.append(locs8)
# locs = locs.append(locs9)
# locs = locs.append(locs10)
# locs = locs.append(locs11)
# locs = locs.append(locs12)
#locs = pd.date_range('2020-09-10', periods=100, freq='1H')
#print(locs)
#print(locs)
#print(locs.iloc[1])


def logic(account, lookback):
    try:
        # Load into period class to simplify indexing
        lookback = helpers.period(lookback)
        
        today = lookback.loc(0) # Current candle
        yesterday = lookback.loc(-1) # Previous candle
        #twoday = lookback.loc(-2)
        #daystartLookback = lookback.loc(-288)
        
        
        # if today['close'] < yesterday['close'] - today['ATR']*3.5:
        if today['close'] < today['MinPoint']:# or today['close'] < yesterday['close'] - (today['ATR']*2) :
        
        #if today['close'] < today['s2'] and today['close'] < today['SMA200'] or (today['WT1'] > (60) and abs(today['WT1'] - today['WT2'] < 5)):# or today['close'] < yesterday['close'] - today['ATR']*3:
            #if yesterday['signal'] == "down":
            exit_price = today['close']
            for position in account.positions:  
                if position.type == 'long':
                    account.close_position(position, 1, exit_price)



        # if today['positions'] == 1.0 and today['close'] > yesterday['close'] + today['ATR']*2:
        #if today['positions'] == 1.0:
        if today['close'] > today['MaxPoint']:
        #if today['close'] > today['r1'] and today['close'] < today['EMA200'] or (today['WT1'] < (-60) and abs(today['WT1'] - today['WT2'] < 5)):# and today['close'] > today['SMA200']:# or today['positions'] == 1.0:
            #print('Today: ' + str(today['r1']))
            #print('Yesterday: ' + str(yesterday['r1']))
            risk          = 1
            entry_price   = today['close']
            entry_capital = account.buying_power*risk
            if entry_capital >= 0 and len(account.positions) == 0:
                account.enter_position('long', entry_capital, entry_price)
     
    except ValueError: 
        pass # Handles lookback errors in beginning of dataset


 # Apply strategy to example
#df = Vortex(df,7)
dfb = TA.PIVOT_FIB(locs)
dfv = df
waveTrend = TA.WTO(dfv)
#print(waveTrend)
#waveTrend = np.insert(waveTrend, 0, 0, axis=0)
df['WT1'] = waveTrend['WT1.']
df['WT2'] = waveTrend['WT2.']
df['r1'] = dfb['r1']
df['r1'] = df['r1'].ffill()
# df['r2'] = dfb['r2']
df['s1'] = dfb['s1']
df['s1'] = df['s1'].ffill()
df['s2'] = dfb['s2']
df['s2'] = df['s2'].ffill()

#print(df.low.nsmallest(5).iloc[-1])

# print(df.iloc[-10:].nsmallest(1, 'low'))
# print(df.iloc[-5:].nlargest(1, 'high'))
nHighest = df.iloc[-5:].nlargest(1, 'high')
print(nHighest['high'])
df['highest'] = nHighest['high']
df['highest'] = df['highest'].ffill()
nLowest = df.iloc[-5:].nsmallest(1, 'low')
df['lowest'] = nHighest['low']
df['lowest'] = df['lowest'].ffill()
# print(df.iloc[-5:])
# df['pivot'] = dfb['pivot']
# df['pivot'] = df['pivot'].ffill()
#df['EMA200'] = ExpMovingAverage(df['close'], 200)
minimaIdxs, maximaIdxs = get_extrema(df['close'], accuracy=20)
#print(minimaIdxs)
#print(df)
print('Calculations Done. Running Backtest')
df['MinPoint'] = df.iloc[minimaIdxs]['low']
df['MinPoint'] = df['MinPoint'].ffill()
df['MaxPoint'] = df.iloc[maximaIdxs]['high']
df['MaxPoint'] = df['MaxPoint'].ffill()
# print('Min Slope: ' + str(pmin))
# print('Max Slope: ' + str(pmax))
# print(mintrend[-1])
# print(minimaIdxs[-1])
#print(df['MinPoint'])
#for i in range(len(locs)):


df.to_csv('DATA.csv')
closep = df['close']
high = df['high']
low = df['low']
openp = df['open']  
trueRanges = []
x=1
while x < len(closep):
    TrueRange = TR(openp[x],high[x],low[x],closep[x],closep[x-1])
    trueRanges.append(TrueRange)
    x += 1
ATR = ExpMovingAverage(trueRanges, 3)
# print(len(closep))
# print(len(ATR))
ATR = np.insert(ATR, 0, 0, axis=0)
df['ATR'] = ATR
print(df)

df = df.reset_index()
# Backtest
backtest = engine.backtest(df)
backtest.start(30, logic)
backtest.results()
backtest.chart()
