import tweepy
from textblob import TextBlob
import csv
import sys
import datetime
import pandas as pd
from scipy import stats
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
from dhooks import Webhook, File
import os
import numpy as np

#Sources: 
#https://www.datasciencecentral.com/profiles/blogs/python-implementing-a-k-means-algorithm-with-sklearn
#https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162259
#http://cs.uef.fi/pages/franti/vq/algorithms.htm


 #Runtime Complexity: O(k * N * I * Tdist)

 #Space Complexity: O(N + k)     ( Store points and centroids)


def twitter(ticker):
    consumer_key = 'FX0OUM3FdSqcbQYUvSFY2LkjV'
    consumer_secret = 'JgYLvIfCWjKeSvKqcEapWuc3G0gYJST6aFfAli2d467VkKsaOW'

    access_token = '1226565917133430784-eUfYQPpTkHBTaSTcHoNEGkV32tif0s'
    access_token_secret = '54mFBBla4b3CQZXeaKLuqT6vg5RRtTJ0MQaaiLH0w1UTR'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # csvFile = open('tweetdata/'+ ticker +'tweets.csv', 'a')

    # fields = ('date','text', 'followers')
    # csvWriter = csv.writer(csvFile, lineterminator= '\n')
    # csvWriter.writerow(['date','text','followers'])

    # geo='40.682563,-100.548699, 1000mi'
    # # try:
    # for tweet in tweepy.Cursor(api.search,q=ticker, lang="en").items(1000):
    #     #print(tweet.created_at, tweet.text)
    #     follower_count = tweet.user.followers_count
    #     #if tweet.created_at >= datetime.datetime(2019,2,24):
    #     csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),follower_count])


    colnames=['date', 'text', 'followers']

    df = pd.read_csv('tweetdata/'+ticker+'tweets.csv',encoding='latin-1', names=colnames, header=None)
    df['followers'][0] = 0.0000
    df['followers'][1] = 0.0000
    df['polarity'] = 0.0000
    df['sentiment_confidence'] = 0.0000

    for index,row in df.iterrows():
        analysis = TextBlob(df['text'][index])
        sentiment, confidence = analysis.sentiment
        df.at[index,'polarity'] = sentiment
        df.at[index,'sentiment_confidence'] = confidence



    dfold = df
    df = df.drop('text', 1)
    df = df.drop('date', 1)
    df_tr = df

    Y = df[['sentiment_confidence']]
    X = df[['polarity']]
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
    count = 0
    for i in score:
        count += 1
        if i > -15:
            print("Optimal Number of Clusters: " + str(count))
            #print(score[count])
            break

    plt.plot(Nc,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()



    # elbow plot showed the point of dropoff to be around 5 clusters

    #Standardize

    clmns = ['followers', 'polarity', 'sentiment_confidence']


    numeric_cols = df_tr.select_dtypes(include=[np.number]).columns
    df_tr_std = df_tr[numeric_cols].apply(zscore)


    #Clustering
    kmeans = KMeans(n_clusters=count, random_state=0).fit(df_tr_std)
    labels = kmeans.labels_

    #Glue back to original data
    df_tr['clusters']=labels
    dfold['clusters']=labels

    clmns.extend(['clusters'])


    #Scatter plot of Wattage and Duration
    sns.lmplot('polarity', 'sentiment_confidence',
            data=df_tr,
            fit_reg=False,
            hue="clusters",
            scatter_kws={"marker": "D",
                            "s": 20})

    #dfold.to_csv('clusterdata/'+ticker+'cluster.csv')
    plt.title('Tweets grouped by Polarity and Subjectivity')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()


inp = input("Type a keyword: ")
twitter(inp)