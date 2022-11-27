#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *
#from implementations import *
#from split_data import * 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import nltk
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

nltk.download('punkt')
from nltk.corpus import stopwords
from collections import Counter
import re
import string
nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('display.max_colwidth',100)


# In[4]:


# A word that is so common that there is no need to use it in a search
ENGLISH_STOP_WORDS = nltk.corpus.stopwords.words('english')

# Adding few extra stop word
ENGLISH_STOP_WORDS = ENGLISH_STOP_WORDS + ['im', 'dont','dunno', 'cant','1', '2', '3', '4', '5', '6', '7'
                                           , '8', '9', "'s", 'u', 'x','user','url','rt','lol']

print(ENGLISH_STOP_WORDS)


# In[5]:


# Calculate the most common words used in the set of all tweets
def get_most_common_words(txt,limit):
    return Counter(txt.split()).most_common()


# In[6]:


# Remove from tweets the punctuation and stop words (= a word that is so common that there is no need to use it in a search.)
def clean_tweet(tweet):
    tweet = "".join([w for w in tweet if w not in string.punctuation])
    tokens = re.split('\W+', tweet)
    tweet = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tweet


# In[7]:


# Change any word belonging to the same word-family into a common word (changing/changes/changed.. ==> change)
def lemmatization(token_tweet):
    tweet = [wn.lemmatize(word) for word in token_tweet]
    return tweet


# In[8]:


# Concatenate the tokennized tweet into a all text like at the beginning
def concatenate(lst):
    concatenate_tweet = ''
    for elem in lst:
        concatenate_tweet = concatenate_tweet + ' ' + elem
    return concatenate_tweet


# In[9]:


def remove_digit(txt):
    txt = ''.join([i for i in txt if not i.isdigit()])
    return txt


# In[10]:


def clean_data(train_df):
    train_df['text'] = train_df['text'].apply(lambda x : clean_tweet(x))
    print("Clean_tweet DONE")
    train_df['text'] = train_df['text'].apply(lambda x : lemmatization(x))
    print("Lemmatization DONE")
    train_df['text'] = train_df['text'].apply(lambda x : concatenate(x))
    print("Concatenate DONE")
    train_df['text'] = train_df['text'].apply(lambda x : clean_tweet(x))
    print("Second clean tweet DONE")
    train_df['text'] = train_df['text'].apply(lambda x : concatenate(x))
    print("Second concatenate DONE")
    train_df['text'] = train_df['text'].apply(lambda x : remove_digit(x))
    print("Remove digit DONE")
    return train_df


# In[ ]:




