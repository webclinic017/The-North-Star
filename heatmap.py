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
from datetime import date 
yf.pdr_override()

hook = Webhook("https://discordapp.com/api/webhooks/752765661201039361/UTV9YdKK9DVzRwAtCmvr6FBbqH_AL02nsJW5WU3KdQMqb2RLIb-feiqd_Yu6o8m8RAV5")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)

ticker_array = []
f = open('heatmapstocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

def percentChange(startPoint, currentPoint):
    return ((currentPoint-startPoint)/startPoint)*100


    #pt_comp = web.DataReader(['BP.L', 'III.L', 'GSK.L', 'OCDO.L', 'RBS.L', 'SVT.L'], 'yahoo',start=start,end=end) ['Adj Close']

for i in range(length):
        tt = ticker_array[i]
        ticker = "{}".format(tt)
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        df = pdr.get_data_yahoo(ticker, period = "5d", interval = "1d", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
        df.to_csv('hourfilesDump/HM' + ticker + '.csv')
        
# pt_comp = pdr.get_data_yahoo(ticker_array, period = "5d", interval = "1d", retry=20, status_forcelist=[404, 429, 500, 502, 503, 504], prepost = True)
# pt_comp.to_csv('HEATMAP.csv')

# current = pt_comp['Adj Close'][4]
# last = pt_comp['Adj Close'][3]
#print(pt_comp)


# array = []
# for i in range(1,length):
#     df = df.sort_values(['Symbol','Date']).reset_index(drop=True)
#     df['Return'] = df.groupby('Symbol')['Open'].pct_change()
#     pct = df['Return']
    
# array.append(pct)
array = []
for i in range(length):

    tt = ticker_array[i]
    ticker = "{}".format(tt)
    df = pd.read_csv('hourfilesDump/HM' + ticker + '.csv')
    df['values'] = 0
    current = df['Adj Close'][4]
    last = df['Adj Close'][3]
    pct = percentChange(last, current)
    array.append(pct)
    
col = []
for val in array:
    if val < 0:
        col.append('red')
    else:
        col.append('green')
    
ticker_names=['ENERGY','FINANCIAL', 'UTILITIES', 'INDUSTRIAL',
        'GOLD MINERS', 'TECH', 'HEALTH CARE', 'CONSUMER DISCRETIONARY',
                 'CONSUMER STAPLES', 'MATERIALS', 'OIL & GAS ', 'U.S. REAL ESTATE',
                  'HOMEBUILDERS', 'CONSTRUCTION', 'REAL ESTATE INDEX FUND',
                   'JUNIOR GOLD MINERS', 'ENERGY', 'METALS & MINING', 'RETAIL', 'SEMICONDUCTOR',
                    'BIOTECH', 'BANK', 'REGIONAL BANKING', 'TELECOM', 'COMMUNICATIONS']

ticker_name = []            
for i in range(len(ticker_names)):
        ticker_name.append(ticker_names[i])
#print(ticker_names)
#ticker_names = (np.asarray(ticker_names)).astype(int)
plt.style.use('dark_background')
bar_plot = plt.barh(ticker_array, array, color=col,height=0.8)
# for index, value in enumerate(ticker_names):
#     plt.text(value, index, str(value))

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        width = rect.get_width()
        # if width > 0:
        plt.text(0,rect.get_y() + rect.get_height()/2,
                ticker_name[idx],
                va='center', ha='center', rotation=0, fontsize='medium',fontweight='bold', fontfamily='monospace', color='#FFF', alpha=0.6)
        # else:
        #     plt.text(width,rect.get_y() + rect.get_height()/2,
        #         ticker_name[idx],
        #         va='center', ha='center', rotation=0, fontweight='bold', color='#FFF')
        

autolabel(bar_plot)

#plt.bar(ticker_names, array, color=col,width=0.8)
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

today = date.today() 
plt.savefig('dailyPics/'+'HM.png', bbox_inches='tight')
discord_pic = File('dailyPics/'+ 'HM.png')
hook.send("Daily Sector Heatmap on " + str(today),file=discord_pic)


#plt.show()
# correlation between stocks 

# corr = pt_rets.corr()

# ax = sns.heatmap(corr)
# plt.imshow(corr, cmap='hot', interpolation='none')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns)
# plt.yticks(range(len(corr)), corr.columns)
# plt.show()