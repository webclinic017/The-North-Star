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

def twitter(ticker):
    hook = Webhook("https://discordapp.com/api/webhooks/746808463685976126/bg5np0vUbjt99nwzM9zm2rKa7XP4LsjYXB-hM9tjhXzRCck-azoyQBLXnSWUpBoNXgUX")

    consumer_key = 'FX0OUM3FdSqcbQYUvSFY2LkjV'
    consumer_secret = 'JgYLvIfCWjKeSvKqcEapWuc3G0gYJST6aFfAli2d467VkKsaOW'

    access_token = '1226565917133430784-eUfYQPpTkHBTaSTcHoNEGkV32tif0s'
    access_token_secret = '54mFBBla4b3CQZXeaKLuqT6vg5RRtTJ0MQaaiLH0w1UTR'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    
    #ticker = str(ticker)


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
    #df = df.drop("followers",axis=1,inplace=True)
    #del df['index']
    #print(df.head())

    df = df.drop('date', 1)
    #df = df.drop(df.index[0], inplace=True)
    print(df.head())

    df_tr = df
    # select proper number of clusters

    # Y = df[['followers']]
    # X = df[['polarity']]
    # Nc = range(1, 20)
    # kmeans = [KMeans(n_clusters=i) for i in Nc]
    # score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
    # plt.plot(Nc,score)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Score')
    # plt.title('Elbow Curve')
    # plt.show()


    # elbow plot showed the point of dropoff to be around 5 clusters

    #Standardize

    clmns = ['followers', 'polarity', 'sentiment_confidence']

    #df_tr_std= stats.zscore(df_tr[clmns])
    numeric_cols = df_tr.select_dtypes(include=[np.number]).columns
    df_tr_std = df_tr[numeric_cols].apply(zscore)
    #df_tr_std = df_tr.apply(zscore)
    #Clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
    labels = kmeans.labels_

    #Glue back to original data
    df_tr['clusters']=labels
    dfold['clusters']=labels

    clmns.extend(['clusters'])

    #print(df_tr[clmns].groupby(['clusters']).mean())
    print(df['clusters'])
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

    #counter = Counter(estimator.labels_)
    clusters_indices = defaultdict(list)
    for index, c  in enumerate(kmeans.labels_):
        clusters_indices[c].append(index)

    cluster1count = len(clusters_indices[0])
    cluster2count = len(clusters_indices[1])
    cluster3count = len(clusters_indices[2])
    # cluster4count = len(clusters_indices[3])
    # cluster5count = len(clusters_indices[4])

    cluster0 = mpatches.Patch(color='#1DA5C3', label= 'Count: '+ str(cluster1count))
    cluster1 = mpatches.Patch(color='#F29328', label='Count: '+str(cluster2count))
    cluster2= mpatches.Patch(color='#35BB53', label='Count: '+str(cluster3count))
    # cluster3= mpatches.Patch(color='red', label='Count: '+str(cluster4count))
    # cluster4= mpatches.Patch(color='red', label='Count: '+str(cluster5count))
    plt.legend(handles=[cluster0, cluster1, cluster2])

    plt.savefig('hourRSIpics/' + ticker + '.png', bbox_inches='tight')
    discord_pic = File('hourRSIpics/' + ticker + '.png')
    hook.send("K-Means Cluster on Keywords: " + ticker, file=discord_pic)
    os.remove('tweetdata/'+ ticker +'tweets.csv')

    #plt.show()

    # public_tweets = api.search('Trump')

    # for tweet in public_tweets:
    #     print(tweet.text)
    #     analysis = TextBlob(tweet.text)
    #     print(analysis.sentiment)
