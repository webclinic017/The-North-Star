import datetime as dt
import grequests
from bs4 import BeautifulSoup
import json
import pandas as pd
from dhooks import Webhook, File
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from collections import Counter
# from sklearn import svm, cross_validation, neighbors
# from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import robin_stocks as r
import matplotlib
import pylab
import os
import sys
import time
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator

def LTSMprediction():
    stock = ('AAL')
    df = pd.read_csv('dailyfilesDump/' + stock + '.csv', skipfooter=30, engine='python')
    df.rename(columns={'date': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
    del df['adjClose']
    del df['adjOpen']
    del df['adjLow']
    del df['adjHigh']
    del df['adjVolume']
    del df['divCash']
    del df['splitFactor']

    df ['Date'] = df['Date'].str.replace('T', ' ', regex=True)
    df ['Date'] = df['Date'].str.replace('Z', '', regex=True)
    df ['Date'] = df['Date'].map(lambda x: str(x)[:-15])
    df ['Date'] = df.index
    #df.index.name = 'Date'
    
    df['Date'] = pd.to_datetime(df['Date'])
    #df['Date'] = df['Date'].apply(mdates.date2num)
    #df = df.astype(float)
    # df.drop_duplicates(subset ="Close", 
    #                 keep = False, inplace = True)
    # df =df[df['Close'] !=0]

    train_set = df.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(train_set)
    print(len(training_set_scaled))
    X_train = []
    y_train = []
    for i in range(60, 526):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0]) 
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.5))
    regressor.add(LSTM(units = 100, return_sequences = False))
    regressor.add(Dropout(0.5))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs=20, batch_size=70, verbose=1, shuffle=False)

    testdataframe = pd.read_csv('dailyfilesDump/' + stock + '.csv', skiprows=range(1,609))
    testdataframe.rename(columns={'date': 'Date', 'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
    testdataframe['Date'] = testdataframe.index
    #print(testdataframe)
    testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
    testdata['Date'] = testdataframe['Date']
    testdata['Open'] = testdataframe['Open']
    testdata['High'] = testdataframe['High']
    testdata['Low'] = testdataframe['Low']
    testdata['Close'] = testdataframe['Close']
    
    real_stock_price = testdata.iloc[:, 1:2].values
    dataset_total = pd.concat((df['Close'], testdata['Close']), axis = 0)
    #print(len(dataset_total))
    inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
    print(inputs)
    pred = inputs[-7:].reshape(1,7,1)
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(60, 90):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
   # print(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test)
    futureElements = []
    #predicted_stock_price = regressor.predict(X_test)
    
   # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # pred = inputs[-7:].reshape(1,7,1)
    print(pred[0])
    #pred = sc.inverse_transform(pred)
    
    #print(pred[-1])
    
    generator = TimeseriesGenerator(X_train, X_train, length=7, batch_size=70)

    model = Sequential()
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (7, 1)))
    model.add(Dropout(0.5))
    model.add(LSTM(units = 100, return_sequences = False))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(generator, epochs=20)

    for i in range(7):
        futureElements.append(model.predict(pred)[0])
        pred = np.append(pred[:,1:,:], [[futureElements[i]]], axis=1)
        
    futureElements = sc.inverse_transform(futureElements)
    
    
    #X_test = np.reshape(1,1,1)
    #step0 = model.predict(X_test)
    #pred = step0[-1].reshape(1,1,1)
    # print(pred[-1])
    # for i in range(7):
    #     futureElements.append(model.predict(pred)[0])
    #     pred = np.append(pred[:,1:,:], [[futureElements[i]]], axis=1)
    # futureElements = sc.inverse_transform(futureElements)

    # futureElement = predicted_stock_price[-1]
    
    # futureElements.append(step1)
    # futureElements.append(step2)
    # futureElements.append(step3)
    # futureElements.append(step4)
    # futureElements.append(step5)
    # futureElements.append(step6)
    # futureElements.append(step7)
    # futureElements = np.array(futureElements)
    # futureElements = np.squeeze(futureElements, axis=1)
    # futureElements = np.squeeze(futureElements, axis=1)
    # futureElements = sc.inverse_transform(futureElements)
    #futureElements = np.squeeze(futureElements, axis=1)
    
    y = []
    y.append(30)
    y.append(31)
    y.append(32)
    y.append(33)
    y.append(34)
    y.append(35)
    y.append(36)
    print(futureElements)
    # futureElements = np.array(futureElements)
    # futureElements = np.reshape(futureElements, (futureElements.shape[0], futureElements.shape[1], 1))
    # for i in range(5):
    #     model.reset_states()
    #     futureElement = model.predict(futureElement)
    #     futureElements.append(futureElement)
    #regressor.fit(X_train, y_train, epochs=2, batch_size=70, verbose=1, shuffle=False)
    # X_test = np.concatenate(([X_test], predicted_stock_price[-1]))
    # predicted_stock_price = regressor.predict(X_test)
    #forecast = predict(inputs, regressor, stock, sc)
    #forecast_dates = predict_dates(num_prediction)

    plt.figure()
    #print(len(real_stock_price))
    #print(predicted_stock_price)
    plt.plot(real_stock_price, color = 'green', label = stock + ' Stock Price')
    #plt.plot(predicted_stock_price, color = 'red', label = 'Trained ' + stock + ' Stock Price')
    plt.plot(y, futureElements, color = 'blue', label = '6 Day Prediction')
    plt.title(stock + ' Price Prediction')
    plt.xlabel('Trading Day')
    plt.ylabel(stock + ' Price')
    plt.legend()
    plt.show()


LTSMprediction()