import tweepy
from textblob import TextBlob
import csv
import sys
import datetime
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

consumer_key = 'FX0OUM3FdSqcbQYUvSFY2LkjV'
consumer_secret = 'JgYLvIfCWjKeSvKqcEapWuc3G0gYJST6aFfAli2d467VkKsaOW'

access_token = '1226565917133430784-eUfYQPpTkHBTaSTcHoNEGkV32tif0s'
access_token_secret = '54mFBBla4b3CQZXeaKLuqT6vg5RRtTJ0MQaaiLH0w1UTR'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

ticker = "AMD"



colnames=['date', 'text', 'followers']

df = pd.read_csv('tweetdata/'+ticker+'tweets.csv',encoding='latin-1', names=colnames, header=None)
df['followers'][0] = 0.0000
df['polarity'] = 0.0000
df['sentiment_confidence'] = 0.0000

for index,row in df.iterrows():
    analysis = TextBlob(df['text'][index])
    sentiment, confidence = analysis.sentiment
    df.at[index,'polarity'] = sentiment
    df.at[index,'sentiment_confidence'] = confidence

df.to_csv('sentimentdata/'+ticker+'sentiment.csv')






df = pd.read_csv('sentimentdata/'+ticker+'sentiment.csv',index_col=0,encoding='latin-1')
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
'''
Y = df[['followers']]
X = df[['polarity']]
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
'''

# elbow plot showed the point of dropoff to be around 5 clusters

#Standardize

clmns = ['followers', 'polarity', 'sentiment_confidence']

df_tr_std= stats.zscore(df_tr[clmns])

#Clustering

kmeans = KMeans(n_clusters=3, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to original data
df_tr['clusters']=labels
dfold['clusters']=labels

clmns.extend(['clusters'])

print(df_tr[clmns].groupby(['clusters']).mean())

#Scatter plot of Wattage and Duration
sns.lmplot('polarity', 'sentiment_confidence',
           data=df_tr,
           fit_reg=False,
           hue="clusters",
           scatter_kws={"marker": "D",
                        "s": 20})

dfold.to_csv('clusterdata/'+ticker+'cluster.csv')
plt.title('Tweets grouped by Polarity and Sentiment Confidence')
plt.xlabel('Polarity')
plt.ylabel('Sentiment_Confidence')
plt.show()

# public_tweets = api.search('Trump')

# for tweet in public_tweets:
#     print(tweet.text)
#     analysis = TextBlob(tweet.text)
#     print(analysis.sentiment)