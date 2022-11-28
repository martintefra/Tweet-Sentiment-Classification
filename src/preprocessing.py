# this file is in charge of preparing the data for the model

#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

pd.set_option('display.max_colwidth',100)

wn = nltk.WordNetLemmatizer()

# A word that is so common that there is no need to use it in a search
ENGLISH_STOP_WORDS = stopwords.words('english')

# Adding few extra stop word
ENGLISH_STOP_WORDS = ENGLISH_STOP_WORDS + ['im', 'dont','dunno', 'cant','1', '2', '3', '4', '5', '6', '7'
                                           , '8', '9', "'s", 'u', 'x','user','url','rt','lol']

# Calculate the most common words used in the set of all tweets
def get_most_common_words(txt,limit):
    return Counter(txt.split()).most_common()[:limit]

# Remove from tweets the punctuation and stop words (= a word that is so common that there is no need to use it in a search.)
def clean_tweet(tweet):
    tweet = "".join([w for w in tweet if w not in string.punctuation])
    tokens = re.split('\W+', tweet)
    tweet = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tweet

# Change any word belonging to the same word-family into a common word (changing/changes/changed.. ==> change)
def lemmatization(token_tweet):
    tweet = [wn.lemmatize(word) for word in token_tweet]
    return tweet

# Concatenate the tokennized tweet into a all text like at the beginning
def concatenate(lst):
    concatenate_tweet = ''
    for elem in lst:
        concatenate_tweet = concatenate_tweet + ' ' + elem
    return concatenate_tweet

def remove_digit(txt):
    txt = ''.join([i for i in txt if not i.isdigit()])
    return txt


def clean_data(df):
    print("Inside clean_data")
    df['text'] = df['text'].apply(lambda x : clean_tweet(x))
    print("Clean_tweet DONE")
    df['text'] = df['text'].apply(lambda x : lemmatization(x))
    print("Lemmatization DONE")
    df['text'] = df['text'].apply(lambda x : concatenate(x))
    print("Concatenate DONE")
    # df['text'] = df['text'].apply(lambda x : clean_tweet(x))
    # print("Second clean tweet DONE")
    # df['text'] = df['text'].apply(lambda x : concatenate(x))
    # print("Second concatenate DONE")
    # df['text'] = df['text'].apply(lambda x : remove_digit(x))
    print("Remove digit DONE")
    return df


