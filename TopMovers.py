from dhooks import Webhook, File
import robin_stocks as r
import json
import pandas as pd
import selenium
import chromedriver_binary
import pandas
import pattern
import fake_useragent
import setuptools
import twine
import unidecode
from bs4 import BeautifulSoup
from newsfetch.news import newspaper
from newsfetch.google import google_search




#Webhook Discord Bot
hook = Webhook("https://discordapp.com/api/webhooks/733027800516263946/fc7y2ZpeMG17sYp3bWAykpb3paDcgxJhw8nXaCBUqAz9gDaJDUhT44zuMCfkdT4ypx7C")
hook2 = Webhook("https://discordapp.com/api/webhooks/733034475302289419/hw2NdT4ZcP1kG70qM7qKbJtcllS4CKn2pOP12p7OVVnluhK0yVfbOaNcFjuyN4FD46Qe")
#Robinhood Login
content = open('robinhood_info.json').read()
config = json.loads(content)
login = r.login(config['username'],config['password'])


def popularityData():
    up_pop = []
    down_pop = []
    # for i in range(length):
    #     print(ticker_array[i])
    #     print(r.stocks.get_popularity(ticker_array[i], info='num_open_positions'))
    #     print(r.stocks.get_fundamentals(ticker_array[i], info='float'))
    up_movers = r.markets.get_top_movers('up', info='symbol')
    down_movers = r.markets.get_top_movers('down', info='symbol')
    hook.send("UPMOVERS: \n")
    for i in range(len(up_movers)):
        pop = r.stocks.get_popularity(up_movers[i], info='num_open_positions')
        hook.send(str(up_movers[i]) + " Popularity: "+ str(pop))

        
    hook.send("DOWNMOVERS: \n")
    for i in range(len(down_movers)):
        pop2 = r.stocks.get_popularity(down_movers[i], info='num_open_positions')
        hook.send(str(down_movers[i]) + " Popularity: "+ str(pop2))
        

popularityData()
#google = google_search('coronavirus', 'finance.yahoo.com/news/' + " after:2020-07-10")
#news = newspaper('https://finance.yahoo.com/news/')
#print(google)
