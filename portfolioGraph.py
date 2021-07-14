from dhooks import Webhook, Embed, File
import robin_stocks as r
import json
import pandas as pd
import time
import datetime as dt
import csv
import matplotlib.pyplot as plt
import matplotlib
import pylab
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

#Robinhood Login
content = open('robinhood_info.json').read()
config = json.loads(content)
login = r.login(config['username'],config['password'])

r.export.export_completed_option_orders(dir_path, file_name=None)
