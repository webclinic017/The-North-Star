import pandas as pd
import json
from tiingo import TiingoClient
import csv

#Changeable Variables
freq = ('1hour')

#TiingoAPI
config = {}
config['session'] = True
config['api_key'] = "56c645789cee9be32a220e9d0a4f6bb84f71ff24"

ticker_array = []
ticker = ('')
def init():
    global ticker
    ticker_array = []
    i=0
    f = open('stocks.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        ticker_array.append(row[0])
        length = len(ticker_array)
        
    print(ticker_array)

    for i in range(length):
        tt = ticker_array[i]
        ticker = "{}".format(tt)
        client = TiingoClient(config)
        ticker_price = client.get_ticker_price(ticker,
                                                fmt='json',
                                                startDate='2020-04-01',
                                                endDate='2020-07-09 ',
                                                frequency=freq)
                                                
        dump = json.dumps(ticker_price, indent=4)
        data=json.loads(dump)
        pd.read_json(dump).to_csv('hourDump/' + ticker + '.csv')
        
init()