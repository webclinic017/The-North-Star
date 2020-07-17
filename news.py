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

hook = Webhook("https://discordapp.com/api/webhooks/733034475302289419/hw2NdT4ZcP1kG70qM7qKbJtcllS4CKn2pOP12p7OVVnluhK0yVfbOaNcFjuyN4FD46Qe")









google = google_search('coronavirus', 'finance.yahoo.com/news/' + " after:2020-07-10")
#news = newspaper('https://finance.yahoo.com/news/')
#print(google)