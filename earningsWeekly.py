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


hook = Webhook("https://discordapp.com/api/webhooks/753854196805271642/sQNxn1pJhQq_zAJPMeA2UCCNo_qG4Gfa_hx2ocVMPLd5VUh4hBx-4dwN8QWjn_ZqK8dT")
with open('earning.png', 'r+b') as f:
    img = f.read()  # bytes

hook.modify(name='Earnings', avatar=img)


yec = YahooEarningsCalendar()
now = datetime.now()
startDate = now.strftime('%b %d %Y %I:%M%p')
#print(startDate)
endDate = now + timedelta(days=7)
endDate = endDate.strftime('%b %d %Y %I:%M%p')
#print(endDate)
startDate = datetime.strptime(
    startDate, '%b %d %Y %I:%M%p')
endDate = datetime.strptime(
    endDate, '%b %d %Y %I:%M%p')


xx = yec.earnings_between(startDate, endDate)
df1 = pd.DataFrame(xx)
df1 = df1[df1.startdatetimetype != 'TAS']
df = df1


def graphChart(df):
    del df['epsactual']
    del df['epssurprisepct']
    del df['timeZoneShortName']
    del df['gmtOffsetMilliSeconds']
    del df['quoteType']
    df.rename(columns={'ticker': 'Ticker', 'companyshortname': 'CompanyShortName', 'startdatetime': 'StartDate', 'startdatetimetype': 'EarningsRelease', 'epsestimate': 'EPS_Estimate'}, inplace=True)

    df['StartDate'] = df['StartDate'].str.replace('T', ' ')
    df['StartDate'] = df['StartDate'].str.replace('Z', ' ')
    df = df[df.EarningsRelease != 'TAS']

    df.dropna(inplace= True)
    #print(df['EPS_Estimate'])

    cm = sns.light_palette("green", as_cmap=True)
    df_styled = df.style.background_gradient(cmap=cm)

    df_styled = df_styled.hide_index()
    dfi.export(df_styled,"EARNINGS.png")
    discord_pic = File("EARNINGS.png")
    hook.send(file=discord_pic)
    os.remove("EARNINGS.png")


hook.send("Upcoming Earnings Releases of the Week")
graphChart(df.head(30))
if len(df) > 30:
    df2 = df[31:60]
    graphChart(df2)
if len(df) > 60:
    df3 = df[61:90]
    graphChart(df3)
if len(df) > 90:
    df4 = df[91:120]
    graphChart(df4)
if len(df) > 120:
    df5 = df[121:150]
    graphChart(df5)