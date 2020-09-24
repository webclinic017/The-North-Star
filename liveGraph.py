import datetime as dt
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.animation as animation
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import pylab
import csv
import time
import os
import sys
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

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

def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow
    
def newData(stock):
    try:
        
        url = 'https://finance.yahoo.com/quote/' + stock
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        # ticker = soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})[0].find('h1').text
        closep = soup.find_all('div', {'class':'D(ib) smartphone_Mb(10px) W(70%) W(100%)--mobp smartphone_Mt(6px)'})[0].find('span').text
        close = closep.replace(',', '')
        # openp = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
        # openp = openp[1].find('span').text
        # openpp = openp.replace(',', '')
        # high = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
        # high = high[4].text
        # result = [x.strip() for x in high.split(' - ')]
        # highp = result[1]
        # highpp = highp.replace(',', '')
        # lowp = result[0]
        # low = lowp.replace(',', '')
        # volumep = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
        # volumep = volumep[6].find('span').text
        # volume = volumep.replace(',', '')
        # tt = [x.strip(')') for x in ticker.split('(') if ')' in x]
        # tick = tt[0]
        
        # print(tick)
        print(close)
        # print(openpp)
        # print(highpp)
        # print(low)
        # print(volume)
        now = dt.datetime.now()
        time = now.strftime('%I:%M%p')
        fieldnames = ["date","close","high","low","open", "volume"]
        toFile(0, close, time, 0, 0, 0, 0, fieldnames)
    except IndexError as e:
        os.execv(sys.executable, ['python3'] + sys.argv)
        print(e)

def toFile(ticker, price_data, time, high, low, openn, volume, fieldnames):  
    with open('liveGraphfiles/BTC.csv', 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            csv_writer.writeheader()

        info = {
            "date": time,
            "close": price_data,
            "high": 0,
            "low": 0,
            "open": 0,
            "volume": 0
                    }

        csv_writer.writerow(info)
  

  
# def update_graph_scatter(n): 
#     X.append(X[-1]+1) 
#     Y.append(Y[-1]+Y[-1] * random.uniform(-0.1,0.1)) 
  
#     data = plotly.graph_objs.Scatter( 
#             x=list(X), 
#             y=list(Y), 
#             name='Scatter', 
#             mode= 'lines+markers'
#     ) 
  
#     return {'data': [data], 
#             'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),yaxis = dict(range = [min(Y),max(Y)]),)} 
  
# if __name__ == '__main__': 
#     app.run_server()


# app = dash.Dash()

# app.layout = html.Div([
#     html.Link(
#             rel='stylesheet',
#             href='https://codepen.io/chriddyp/pen/bWLwgP.css'
#         ),
#     dcc.Input(id='input-box', value='', type='text', placeholder='Enter a Stock symbol', ),
#     html.Button('Submit', id='button'),
#     html.Div(),
#     html.P('5 Calls Per Min'),
#     dcc.Graph(
#         id='candle-graph', animate=True, style={"backgroundColor": "#1a2d46", 'color':'#ffffff'},),
#     html.Div([
#             html.P('Developed by: ', style={'display': 'inline', 'color' : 'white'}),
#             html.A('Austin Kiese', href='http://www.austinkiese.com'),
#             html.P(' - ', style={'display': 'inline', 'color' : 'white'}),
#             html.A('cryptopotluck@gmail.com', href='mailto:cryptopotluck@gmail.com')
#         ], className="twelve columns",
#             style={'fontSize': 18, 'padding-top': 20}

#         )
# ])



# @app.callback(Output('candle-graph', 'figure'),
#               [Input('button', 'n_clicks')],
#               [State('input-box', 'value')])

# def update_layout(self):

#     #Getting Dataframes Ready
#     newData('BTC-USD')

#     df = pd.read_csv('liveGraphfiles/BTC.csv')



#     BuySide = go.Candlestick(
#         x=df.index,
#         open=df['open'],
#         high=df['high'],
#         low=df['low'],
#         close=df['close'],
#         increasing={'line': {'color': '#00CC94'}},
#         decreasing={'line': {'color': '#F50030'}},
#         name='candlestick'
#     )
#     data = [BuySide]

#     layout = go.Layout(
#         paper_bgcolor='#27293d',
#         plot_bgcolor='rgba(0,0,0,0)',

#     )


#     return {'data': data, 'layout' : layout}

xcol = deque(maxlen = 20) 
xcol.append(1) 
  
# Y = deque(maxlen = 20) 
# Y.append(1) 
#xcol = []
ycol = []
app = dash.Dash() 
  
app.layout = html.Div( 
    [ 
        dcc.Graph(id = 'live-graph', animate = True), 
        dcc.Interval( 
            id = 'graph-update', 
            interval = 2000, 
            n_intervals = 0
        ), 
    ] 
) 
  
@app.callback( 
    Output('live-graph', 'figure'), 
    [ Input('graph-update', 'n_intervals') ] 
) 

def update_graph_scatter(n): 

    url = 'https://finance.yahoo.com/quote/BTC-USD'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    # ticker = soup.find_all('div', {'class':'D(ib) Mt(-5px) Mend(20px) Maw(56%)--tab768 Maw(52%) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)'})[0].find('h1').text
    closep = soup.find_all('div', {'class':'D(ib) smartphone_Mb(10px) W(70%) W(100%)--mobp smartphone_Mt(6px)'})[0].find('span').text
    close = closep.replace(',', '')
    ycol.append(close)
    # newData('BTC-USD')

    # df = pd.read_csv('liveGraphfiles/BTC.csv')
    # now = dt.datetime.now()
    # time = now.strftime('%I:%M%p')
    # xcol.append(time)
    xcol.append(xcol[-1]+1) 
    #print(xcol)
    #print(ycol)
    # data = plotly.graph_objs.Scatter( 
    #         x=list(xcol), 
    #         y=list(ycol), 
    #         name='Scatter', 
    #         mode= 'lines+markers'
    # ) 
    fig = go.Figure(data=[go.Scatter(x=list(xcol), y=list(ycol))])

    layout = go.Layout(xaxis=dict(range=[min(xcol),max(xcol)]),yaxis=dict(range=[min(ycol),max(ycol)]))
    #app.layout = layout

    
    #time.sleep(2)
    return fig



if __name__ == '__main__':
    app.run_server(port=8085, debug=True)



    # def update_graph_scatter(n): 

    # data = plotly.graph_objs.Scatter( 
    #         x=df['date'], 
    #         y=df['close'], 
    #         name='Scatter', 
    #         mode= 'lines+markers'
    # ) 

    # layout = go.Layout(
    #     paper_bgcolor='#27293d',
    #     plot_bgcolor='rgba(0,0,0,0)',

    # )
  
    # return {'data': [data], 
    #         'layout' : layout} 