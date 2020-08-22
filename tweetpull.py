import tweepy
from textblob import TextBlob
import csv
import sys
import datetime
import pandas as pd
import discord

def pull(ticker):
    consumer_key = 'FX0OUM3FdSqcbQYUvSFY2LkjV'
    consumer_secret = 'JgYLvIfCWjKeSvKqcEapWuc3G0gYJST6aFfAli2d467VkKsaOW'

    access_token = '1226565917133430784-eUfYQPpTkHBTaSTcHoNEGkV32tif0s'
    access_token_secret = '54mFBBla4b3CQZXeaKLuqT6vg5RRtTJ0MQaaiLH0w1UTR'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    ticker = str(ticker)

    #Open/Create a file to append data
    csvFile = open('tweetdata/'+ ticker +'tweets.csv', 'a')
    #Use csv Writer
    fields = ('date','text', 'followers')
    csvWriter = csv.writer(csvFile, lineterminator= '\n')
    csvWriter.writerow(['date','text','followers'])

    try:
        for tweet in tweepy.Cursor(api.search,q=ticker, lang="en", result_type="popular").items():
            print(tweet.created_at, tweet.text)
            follower_count = tweet.user.followers_count
            #if tweet.created_at >= datetime.datetime(2019,2,24):
            csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),follower_count])
    except Exception as e:
        print(e)
        pass