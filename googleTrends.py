from pytrends.request import TrendReq
import pandas as pd
import csv

ticker_array = []
f = open('stocks.csv')
csv_f = csv.reader(f)
for row in csv_f:
    ticker_array.append(row[0])
    length = len(ticker_array)

pytrends = TrendReq(hl='en-US', tz=360, retries=10, backoff_factor=0.5)
#pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})

kw_list = ["TSLA"]
pytrends.build_payload(kw_list, cat=0, timeframe='now 1-H', geo='US', gprop='')
hourStart = 16
pd = pytrends.get_historical_interest(kw_list, year_start=2020, month_start=7, day_start=23, hour_start=20, year_end=2020, month_end=7, day_end=23, hour_end=20, cat=0, geo='US', gprop='', sleep=0)

print(pd)