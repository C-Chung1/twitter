import requests
import json
import tweepy
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import os
import string
import re

consumer_key= os.environ.get('CONSUMER_KEY')
consumer_secret= os.environ.get('CONSUMER_SECRET')
access_token= os.environ.get('ACCESS_TOKEN')
access_token_secret= os.environ.get('ACCESS_TOKEN_SECRET')
bearer_token = os.environ.get("BEARER_TOKEN")

stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))
sentiments = SentimentIntensityAnalyzer()

auth= tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)


# figure out how to pull from more than 1 user.

def get_user_tweets(userid):
    
    tweets = []
    likes = []
    time = []
    retweets = []
    
    user_tweets = tweepy.Cursor(api.user_timeline, id=userid,tweet_mode = 'extended').items(100)
    
    for i in user_tweets:
        tweets.append(i.full_text)
        retweets.append(i.retweet_count)
        try:
            likes.append(i.retweeted_status.favorite_count)
        except:
            likes.append(i.favorite_count)
        time.append(i.created_at)

    df = pd.DataFrame({'tweets': tweets, 'likes':likes, 'time':time, 'id':userid, 'retweets':retweets})

    return df

# cleans tweets to normalize
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('rt', '', text, 1)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

# finish the republican list
republican = ['SenShelby', 'Ttuberville', 'SenDanSullivan', 'lisamurkowski','SenTomCotton',
              'JohnBoozman', 'marcorubio', 'ScottforFlorida', 'SenatorRisch', 'MikeCrapo',
              'Braun4Indiana', 'SenToddYoung', 'ChuckGrassley','SenJoniErnst', 'RogerMarshallMD',
              'JerryMoran', 'RandPaul','McConnellPress','JohnKennedyLA','BillCassidy',
              'SenatorCollins', 'SenatorWicker', 'SenHydeSmith', 'HawleyMO', 'RoyBlunt',
              'SteveDaines', 'SenSasse', 'SenatorFischer', 'SenThomTillis', 'SenatorBurr',
              'SenJohnHoeven', 'SenKevinCramer', 'senrobportman', 'SenatorLankford', 'InhofePress',
              'SenToomey', 'SenatorTimScott', 'LindsayGrahamSC', 'SenJohnThune', 'SenatorRounds',
              'MarshaBlackburn', 'BillHagertyTN', 'SenTedCruz', 'JohnCornyn', 'SenMikeLee',
              'SenatorRomney', 'SenCapito', 'SenRonJohnson', 'SenLummis', 'SenJohnBarrasso']

republican_df = pd.DataFrame(columns = ['tweets', 'likes', 'time', 'id', 'retweets'])


for i in republican:
    x = get_user_tweets(i)
    republican_df = republican_df.append(x)
    
democrat = ['SenMarkKelly', 'SenatorSinema', 'SenFeinstein', 'AlexPadilla4CA', 'MichaelBennet',
            'SenatorHick', 'ChrisMurphyCT', 'SenBlumenthal', 'SenCoonsOffice', 'SenatorCarper',
            'ossoff', 'ReverendWarnock', 'SenBrianSchatz', 'maziehirono', 'SenDuckworth',
            'SenatorDurbin', 'ChrisVanHollen', 'SenatorCardin', 'SenWarren', 'SenMarkey',
            'SenStabenow', 'SenGaryPeters', 'amyklobuchar', 'SenTinaSmith', 'SenatorTester',
            'SenJackyRosen', 'SenCortezMasto', 'SenatorShaheen', 'Maggie_Hassan', 'SenatorMenendez',
            'CoryBooker', 'SenatorLujan', 'MartinHeinrich', 'SenSchumer', 'SenGillibrand',
            'SenSherrodBrown', 'RonWyden', 'SenJeffMerkley', 'SenBobCasey', 'SenWhitehouse',
            'SenJackReed', 'SenatorLeahy', 'MarkWarner', 'timkaine', 'PattyMurray',
            'SenatorCantwell', 'Sen_JoeManchin', 'SenatorBaldwin', 'SenAngusKing', 'SenSanders'
            ]

democrat_df = pd.DataFrame(columns = ['tweets', 'likes', 'time', 'id', 'retweets'])

for i in democrat:
    x = get_user_tweets(i)
    democrat_df = democrat_df.append(x)

# EDA
sns.distplot(democrat_df['likes'])
sns.distplot(republican_df['likes'])
sns.distplot(democrat_df['retweets'])
sns.distplot(republican_df['retweets'])

# Clean democrat DF
democrat_df['tweets_2'] = democrat_df['tweets'].apply(clean)

democrat_df['tweets'].head()

democrat_df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in democrat_df["tweets_2"]]
democrat_df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in democrat_df["tweets_2"]]
democrat_df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in democrat_df["tweets_2"]]

x = sum(democrat_df["Positive"])
y = sum(democrat_df["Negative"])
z = sum(democrat_df["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print("Negative")
    else:
        print("Neutral")
sentiment_score(x, y, z)

print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)

# Clean republican DF
republican_df['tweets_2'] = republican_df['tweets'].apply(clean)

republican_df['tweets'].head()

republican_df["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in republican_df["tweets_2"]]
republican_df["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in republican_df["tweets_2"]]
republican_df["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in republican_df["tweets_2"]]

a = sum(republican_df["Positive"])
b = sum(republican_df["Negative"])
c = sum(republican_df["Neutral"])

sentiment_score(a, b, c)

print("Positive: ", a)
print("Negative: ", b)
print("Neutral: ", c)

grouped_D = democrat_df.groupby('id')
grouped_R = republican_df.groupby('id')

# Calculating sentiment of each individual Senator

d = grouped_D['Positive'].sum().to_frame()
e = grouped_D['Negative'].sum().to_frame()
f = grouped_D['Neutral'].sum().to_frame()


g = grouped_R['Positive'].sum().to_frame()
h = grouped_R['Negative'].sum().to_frame()
i = grouped_R['Neutral'].sum().to_frame()

g = g.join(h, how='left')
g = g.join(i, how='left')

# Creating levels of positivity
g['sentiment'] = np.where(g['Positive'] > g['Negative'], 'Positive', 'Negative')
g['sentiment'] = np.where(g['Positive'] >(g['Negative'] * 2), 'Very Positive', g['sentiment'])
g['sentiment'] = np.where(((g['Negative'] * 1.1) > g['Positive']), 'Barely Positive', g['sentiment'])

d = d.join(e, how='left')
d = d.join(f, how='left')

d['sentiment'] = np.where(d['Positive'] > d['Negative'], 'Positive', 'Negative')
d['sentiment'] = np.where(d['Positive'] >(d['Negative'] * 2), 'Very Positive', d['sentiment'])
d['sentiment'] = np.where(((d['Negative'] * 1.1) > d['Positive']), 'Barely Positive', d['sentiment'])

save_location = 'C:\\Users\\chanb\\OneDrive\\Desktop\\Twitter Wordclouds'

for name, group in grouped_D:
  text = group['tweets_2']
  wordcloud = WordCloud().generate(str(text))

  plt.title(name)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.ioff()
  plt.savefig(os.path.join(save_location,'Democrat ') + name + '.png')


for name, group in grouped_R:
  text = group['tweets']
  wordcloud = WordCloud(stopwords = stopword).generate(str(text))

  plt.title(name)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.ioff()
  plt.savefig(os.path.join(save_location,'Republican ') + name + '.png')
 