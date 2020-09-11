from dhooks import Webhook, Embed, File
import robin_stocks as r
import json
import pandas as pd
import time
import datetime as dt
import csv
import matplotlib.pyplot as plt
import matplotlib
import urllib as u
import pylab
import numpy as np
from pandas_datareader import data as pdr
from yahoo_earnings_calendar import YahooEarningsCalendar
from bs4 import BeautifulSoup as bs
import warnings
from datetime import datetime
import sys
from datetime import timedelta
import dataframe_image as dfi
from dhooks import Webhook, Embed, File
import os
import seaborn as sns


warnings.filterwarnings("ignore")

hook = Webhook("https://discordapp.com/api/webhooks/753854196805271642/sQNxn1pJhQq_zAJPMeA2UCCNo_qG4Gfa_hx2ocVMPLd5VUh4hBx-4dwN8QWjn_ZqK8dT")
with open('lord.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='TheMeciah', avatar=img)


yec = YahooEarningsCalendar()
now = datetime.now()
startDate = now.strftime('%b %d %Y %I:%M%p')
print(startDate)
endDate = now + timedelta(days=1)
endDate = endDate.strftime('%b %d %Y %I:%M%p')
print(endDate)
startDate = datetime.strptime(
    startDate, '%b %d %Y %I:%M%p')
endDate = datetime.strptime(
    endDate, '%b %d %Y %I:%M%p')


content = open('robinhood_info.json').read()
config = json.loads(content)
login = r.login(config['username'],config['password'])

# getEarningsMovers = r.markets.get_all_stocks_from_market_tag('upcoming-earnings', info='symbol')
# print(getEarningsMovers)
# earnings = r.stocks.get_earnings("KR", info=None)

# df = pd.DataFrame(earnings)
# df.to_csv('dailyfilesDump/EARNINGS2.csv')
# print(df['eps'])

# earn = yec.get_earnings_of('AAPL')
# df = pd.DataFrame(earn)
# df.to_csv('dailyfilesDump/EARNINGS.csv')

xx = yec.earnings_between(startDate, endDate)
df = pd.DataFrame(xx)

del df['epsactual']
del df['epssurprisepct']
del df['timeZoneShortName']
del df['gmtOffsetMilliSeconds']
del df['quoteType']
df.rename(columns={'ticker': 'Ticker', 'companyshortname': 'CompanyShortName', 'startdatetime': 'StartDate', 'startdatetimetype': 'EarningsRelease', 'epsestimate': 'EPS_Estimate'}, inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.index.name = '#'
df['StartDate'] = df['StartDate'].str.replace('T', ' ')
df['StartDate'] = df['StartDate'].str.replace('Z', ' ')
#html_table = df.to_html(index = False)
# html_table = HTML(df.to_html(index = False))
# print(html_table)
# cols = df.columns.tolist()
# print(cols)
cm = sns.light_palette("green", as_cmap=True)
df_styled = df.style.background_gradient(cmap=cm)
#df_styled = df_styled.hide_index().render()
# blankIndex = [''] * len(df)
#df_styled.index = blankIndex
df_styled = df_styled.hide_index()
dfi.export(df_styled,"EARNINGS.png")
discord_pic = File("EARNINGS.png")
hook.send("Upcoming Earnings Releases of the Day",file=discord_pic)
os.remove("EARNINGS.png")
# for x in xx:
#     print("start--->{}".format(x['startdatetime']))
#     print("ticker--->{}".format(x['ticker']))
#     print("epsestimate--->{}".format(x['epsestimate']))
#     print("epsactual--->{}".format(x['epsactual']))
#     print("epssurprisepct--->{}".format(x['epssurprisepct']))

# def get_eps_estimates(url):
#     try:
#         html_source = u.request.urlopen(url).read()
#         soup = bs(html_source, 'lxml')
#         # 
#         # table
#         table = soup.find_all('table', attrs={'class': 'yfnc_tableout1'})
#         header = [th.text for th in table[0].find_all(class_='yfnc_tablehead1')]
#         header_title = header[0]
#         header_cols = header[1:5]
#         index_row_labels = header[-5:]
#         body = [[td.text for td in row.select('td')] for row in table[0].find_all('tr')]
#         body = body[1:]
#         df = pd.DataFrame.from_records(body)
#         df = df.ix[:, 1:]
#         df.index = index_row_labels
#         header_cols = pd.Series(header_cols)
#         header_cols = header_cols.str.replace(
#             'Year', 'Year ').str.replace('Qtr.', 'Qtr. ')
#         df.columns = header_cols
#         eps_est = df.convert_objects(convert_numeric=True)
#     except IndentationError as e:
#         print(e)
#     return eps_est

# symbol = 'SWKS'
# ticker = "{}".format(symbol)
# base_url = ('https://finance.yahoo.com/quote/' + symbol + '/analysis?ltr=1')
# est = get_eps_estimates(base_url)
# print(est)



