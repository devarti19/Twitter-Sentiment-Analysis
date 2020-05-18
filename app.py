#Importing libraries
import tweepy
import numpy as np
from flask import Flask, request, jsonify, render_template,url_for, redirect
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# Variables that contains the credentials to access Twitter API
# These credentials are unique to every account
ACCESS_TOKEN = '1014097837381111808-1ZmyTn9NwXhnlnH3rLJdjLg7WmwFTG'
ACCESS_SECRET = 'siiC7b529NpcGT8uwopjOaauSRZoUQdlN38mLL0gX2TLK'
CONSUMER_KEY = 'ZSTjRgpSwrcpgLDqTug1tnHVS'
CONSUMER_SECRET = '2mfFBHGIV0OT7b5LbgWrAbBhoyr3tB7GiGTtCGiz0pN8S9EFKv'

# Setup access to API

# OAuthHandler allows to check the credentials
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# set_access_token is used to create a session
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)


# Each tweet is considered as a document so each tweet will occupy a record in the DataFrame
# More attributes regarding a tweet can be accessed through the following link:
# https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object

def extract_tweet_attributes(tweet_object):
    # create empty list
    tweet_list =[]
    # looping through the tweet object
    # Loop Starts
    for tweet in tweet_object:
        tweet_id = tweet.id                   # unique integer identifier for tweet
        text = tweet.text
        favorite_count = tweet.favorite_count # Hearts or likes on a particular tweet
        retweet_count = tweet.retweet_count   # Number of times a particular tweet has been retweeted
        created_at = tweet.created_at         # utc time tweet created
        source = tweet.source                 # utility used to post tweet whether an iphone or an android
        reply_to_status = tweet.in_reply_to_status_id # if reply int of orginal tweet id
        reply_to_user = tweet.in_reply_to_screen_name # if reply original tweetes screenname
        retweets = tweet.retweet_count        # number of times this tweet retweeted
        place = tweet.user.location                     # utf-8 text or description of the tweet
        # append attributes to list
        tweet_list.append({'tweet_id':tweet_id,
                          'Text':text,
                          'favorite_count':favorite_count,
                          'retweet_count':retweet_count,
                          'created_at':created_at,
                          'source':source,
                          'reply_to_status':reply_to_status,
                          'reply_to_user':reply_to_user,
                          'retweets':retweets,
                          'Place':place
                          })
        #Loop Ends
    # create dataframe
    tweet_dataframe = pd.DataFrame(tweet_list, columns=['tweet_id', 'Text','favorite_count',
                                           'retweet_count',
                                           'created_at',
                                           'source',
                                           'reply_to_status',
                                           'reply_to_user',
                                           'retweets',
                                           'Place'])
    # returning the DataFrame formed
    return tweet_dataframe

def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"n\'t", "n not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


app = Flask(__name__)

@app.route("/")
def home():
   return render_template('home.html',len=0, tweet=[], classify=[])

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    # Create API object
    if request.method == "POST":

        tweets = api.user_timeline(screen_name=request.form.values(), count=10)  # Here we can use hashtags or usernames
        # tweet_df is the DataFrame which will store all the tweets info
        tweet_df = extract_tweet_attributes(tweets)
        # Removing the names of the people who have retweeted- names are given as @name
        tweet_df['clean_text'] = tweet_df['Text'].replace(r'@\w+','',regex=True)
        #remove apostrphe from the word
        tweet_df['clean_text']= [decontracted(i) for i in tweet_df['clean_text']]
        # Removing the word RT which signifies retweeted from the clean text
        tweet_df['clean_text'].replace('RT','',inplace=True, regex=True)
        # Removing the links present in our clean text
        tweet_df['clean_text'].replace('((www\.[^\s]+)|(https?://[^\s]+))', '\0', inplace=True, regex=True)
        #Removing special characters, numbers, punctuations as none of them would add any value while making sentiment analysis
        tweet_df['clean_text'].replace('[^a-zAA-Z]+', ' ', inplace=True, regex=True)

        # Converting all the clean_text column into lower case
        tweet_df['clean_text'] = tweet_df['clean_text'].str.lower()

        # TOKENIZATION # Splitting each row of clean_text column into list of words
        tweet_df['clean_text'] = tweet_df['clean_text'].str.split()

        # Getting list of stopwords present in the nltk library
        stop_words = stopwords.words('english')
        stop_words.remove('no')
        stop_words.remove('not')

        # Removing the stopwords from every row of the column 'clean_text'
        tweet_df['clean_text'] = tweet_df['clean_text'].apply(lambda x:[item for item in x if item not in stop_words])

        # Joining all the tokens into a string
        tweet_df['clean_text'] = tweet_df['clean_text'].apply(lambda x: ' '.join(x))

        # Doing tokenization through WhitespaceTokenizer
        w_s_tokenizer = WhitespaceTokenizer()
        tweet_df['clean_text'] = tweet_df['clean_text'].apply(lambda x: w_s_tokenizer.tokenize(x))

        # Doing Lemmatization through WordNetLemmatizer
        w_n_lemmatizer = WordNetLemmatizer()
        tweet_df['clean_text'] = tweet_df['clean_text'].apply(lambda x: [w_n_lemmatizer.lemmatize(item) for item in x])

        # Joining all the tokens into a string
        tweet_df['clean_text'] = tweet_df['clean_text'].apply(lambda x: ' '.join(x))
        t=tweet_df['Text']

        classify= []
        pos=[]
        neg=[]
        for tweet in tweet_df['clean_text']:
            analyse = TextBlob(tweet, analyzer=NaiveBayesAnalyzer())
            classify.append(analyse.sentiment[0])
            pos.append(round(analyse.sentiment[1],2))
            neg.append(round(analyse.sentiment[2],2))
        classify= pd.Series(classify)

        pos= pd.Series(pos)
        neg= pd.Series(neg)
        df = pd.concat([classify, pos, neg], axis=1)
        df.columns = ['Classification', 'Positive', 'Negative']
        a = pd.DataFrame(df['Classification'].value_counts())
        a.columns = ['counts']

        #Plot bar plot for positive and negative
        a = sns.barplot(x=a.index, y=a['counts'], palette='vlag')
        fig = a.get_figure()
        fig.savefig('static/images/plot1.png')

        #To plot polarity graph
        width = 0.35  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots(figsize=(20, 10))

        ax.bar(df.index, df['Positive'], width, label='Positive', color='c')
        ax.bar(df.index, df['Negative'], width, bottom=df['Positive'], label='Negative', color='r')

        ax.set_ylabel('Sentiment')
        ax.set_title('Sentiment for each sentence')
        ax.legend()
        fig.savefig('static/images/plot2.png')
        classify = classify.replace({'pos': 'Positive Tweet', 'neg': 'Negative Tweet'})

        return render_template('Prediction.html', len=len(tweets), tweet=t, classify=classify,url1='static/images/plot1.png',url2='static/images/plot2.png')

if __name__ == "__main__":
    app.run(debug=True)