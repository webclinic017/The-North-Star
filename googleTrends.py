import pytrends
from pytrends.request import TrendReq
import pandas as pd
import time
import datetime
from datetime import datetime, date, time
import csv
import matplotlib.pyplot as plt
import seaborn as sns

ticker_array = []
f = open('stocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

# pytrends = TrendReq(hl='en-US')
# #pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})

# kw_list = ["FE", "FE sell", "FE buy"]
# pytrends.build_payload(kw_list, cat=0, timeframe='now 1-H', geo='US', gprop='')
# hourStart = 16
# pd = pytrends.get_historical_interest(kw_list, year_start=2020, month_start=7, day_start=24, hour_start=0, year_end=2020, month_end=7, day_end=24, hour_end=23, cat=0, geo='US', gprop='', sleep=0)

# print(pd)



def trendsAnalysis():
    pytrends = TrendReq(hl='en-US', tz=180)
    kw_list = ["TSLA", "FE", "AAPL"]
    sns.set(color_codes=True)
    pd = pytrends.get_historical_interest(kw_list, year_start=2020, month_start=7, day_start=24, hour_start=0, year_end=2020, month_end=7, day_end=24, hour_end=23, cat=0, geo='US', gprop='news', sleep=0)
    dx = pd.plot.line(figsize = (9,6), title = "Interest Over Time")
    dx.set_xlabel('Date')
    dx.set_ylabel('Trends Index')
    dx.tick_params(axis='both', which='major', labelsize=13)
    plt.show()

trendsAnalysis()