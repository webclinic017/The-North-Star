import pandas as pd
import seaborn as sns
import datetime as dt
import pandas as pd
from dhooks import Webhook, File
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
import csv
yf.pdr_override()

ticker_array = []
f = open('heatmapstocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

def percentChange(startPoint, currentPoint):
    return ((currentPoint-startPoint)/startPoint)*100


    #pt_comp = web.DataReader(['BP.L', 'III.L', 'GSK.L', 'OCDO.L', 'RBS.L', 'SVT.L'], 'yahoo',start=start,end=end) ['Adj Close']

# pt_comp = pdr.get_data_yahoo(ticker_array, period = "5d", interval = "1d", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
# pt_comp.to_csv('HEATMAP.csv')

# current = pt_comp['Adj Close'][4]
# last = pt_comp['Adj Close'][3]
#print(pt_comp)
df = pd.read_csv('HEATMAP.csv')
array = []
for i in range(1,length):
    df = df.sort_values(['Symbol','Date']).reset_index(drop=True)
    df['Return'] = df.groupby('Symbol')['Open'].pct_change()
    pct = df['Return']
    
array.append(pct)

# # pct = percentChange(current, last)

# plt.bar(array, ticker_array, width=0.4)
# plt.show()
#correlation between stocks 

# corr = pt_rets.corr()

# ax = sns.heatmap(corr)
# plt.imshow(corr, cmap='hot', interpolation='none')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns)
# plt.yticks(range(len(corr)), corr.columns)
# plt.show()