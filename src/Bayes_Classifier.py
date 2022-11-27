#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *
#from implementations import *
#from split_data import * 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pandas as pd


# In[2]:


#from simpletransformers.classification import ClassificationModel, ClassificationArgs
#import pandas as pd


# In[3]:


# Main external library : Natural Language Toolkit (nltk)
import nltk
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()

nltk.download('punkt')
from nltk.corpus import stopwords


# # Load positive tweets & remove duplicate

# In[4]:


train_pos_class = []


# In[5]:


filename1 = 'data/train_pos.txt'
with open(filename1) as f1:
    train_pos = f1.readlines()
len(train_pos)


# In[6]:


#Remove duplciate positive tweets
train_pos = list(dict.fromkeys(train_pos))
len(train_pos)


# In[7]:


#Put each positive tweet in lower case in its own list with correct label (1)
for line in train_pos:
    train_pos_class.append([line.lower(),1])


# In[8]:


train_pos_class


# # Load negative tweets & remove duplicate

# In[9]:


train_neg_class = []


# In[10]:


filename2 = 'data/train_neg.txt'
with open(filename2) as f2:
    train_neg = f2.readlines()
len(train_neg)


# In[11]:


#Remove duplciate negative tweets
train_neg = list(dict.fromkeys(train_neg))
len(train_neg)


# In[12]:


#Put each negative tweet in lower case in its own list with correct label (0)
for line in train_neg:
    train_neg_class.append([line.lower(),0])


# In[13]:


train_neg_class


# # Cleaning Data

# In[14]:


from collections import Counter
import re
import string
nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('display.max_colwidth',100)


# In[15]:


#Concatenate the two training sets of positive and negative tweets
for line in train_neg_class:
    train_pos_class.append(line)


# In[16]:


train_data = train_pos_class
train_data


# In[17]:


train_df = pd.DataFrame(train_data)

#Labeling
train_df.columns = ["text", "cat_label"]


# In[18]:


len(train_df)


# In[19]:


train_df.head()


# In[20]:


from preprocessing import *


# In[21]:


whole_tweets = " ".join(train_df.text.values)
get_most_common_words(whole_tweets,10)


# In[22]:


train_df = clean_data(train_df)


# # Naïve Bayes Classifier

# In[23]:


train_neg_class = train_df.loc[train_df['cat_label'] == 0]
train_pos_class = train_df.loc[train_df['cat_label'] == 1]
print(len(train_df))
print(len(train_pos_class))
print(len(train_neg_class))


# In[24]:


whole_neg = " ".join(train_neg_class.text.values)
word_neg_rep = get_most_common_words(whole_neg,10)
word_neg_rep


# In[25]:


whole_pos = " ".join(train_pos_class.text.values)
word_pos_rep = get_most_common_words(whole_pos,10)
word_pos_rep


# In[26]:


lst_pos = [x[0] for x in word_pos_rep]
lst_neg = [x[0] for x in word_neg_rep]
lst_pos = lst_pos[0:20]
lst_neg = lst_neg[0:20]
list(set(lst_neg).intersection(lst_pos))


# In[27]:


sum_pos = 0
sum_neg = 0

for (word, apparition) in word_pos_rep:
    sum_pos = sum_pos + apparition

for (word, apparition) in word_neg_rep:
    sum_neg = sum_neg + apparition


# In[28]:


def proba_word_good(word):
    for i in range(len(word_pos_rep)):
        if(word == word_pos_rep[i][0]):
            return word_pos_rep[i][1]/sum_pos
    return 1/sum_pos
    
def proba_word_bad(word):
    for i in range(len(word_neg_rep)):
        if(word == word_neg_rep[i][0]):
            return word_neg_rep[i][1]/sum_neg
    return 1/sum_neg


# In[29]:


def posteriori_probability(text, posteriori_prob_good=1, posteriori_prob_bad=1):
    text_df = pd.DataFrame([text])
    text_df.columns = ["text"]
    word_text = " ".join(text_df.text.values)
    word_distribution = get_most_common_words(word_text,10)
    
    for (word,apparition) in word_distribution:
        posteriori_prob_good = posteriori_prob_good * proba_word_good(word)
        posteriori_prob_bad = posteriori_prob_bad * proba_word_bad(word)
        
    prob_good = len(train_pos_class)/len(train_df)
    prob_bad = len(train_neg_class)/len(train_df)
    
    return (posteriori_prob_good*prob_good/(posteriori_prob_good*prob_good+posteriori_prob_bad*prob_bad))


# # Test tweet

# In[30]:


filename3 = 'data/test_data.txt'
with open(filename3) as f3:
    test_data = f3.readlines()
len(test_data)


# In[31]:


#Remove duplciate negative tweets
test_data = list(dict.fromkeys(test_data))
len(test_data)


# In[32]:


for i in range(len(test_data)):
    test_data[i] = test_data[i][2:]


# In[33]:


test_data


# In[34]:


test_df = pd.DataFrame(test_data)


# In[35]:


test_df


# In[36]:


test_class = []

for line in test_data:
    test_class.append([line.lower()])

test_df.columns = ['text']
test_df


# In[37]:


test_df = clean_data(test_df)
test_df


# In[38]:


y_pred = []
for i in range(len(test_df)):
    print(i)
    if (posteriori_probability(test_df['text'][i]) >= 0.5):
        y_pred.append(1)
    else:
        y_pred.append(-1)  


# In[39]:


len(test_df)


# In[40]:


test_data_df = pd.DataFrame(y_pred)


# In[41]:


test_data_df['Id'] = range(1, len(test_data_df) + 1)


# In[42]:


test_data_df.columns = ["Id", "Prediction"]


# In[43]:


columns_titles = ["Prediction","Id"]
test_data_df=test_data_df.reindex(columns=columns_titles)
test_data_df.columns = ["Id", "Prediction"]


# In[44]:


test_data_df


# In[45]:


test_data_df.to_csv("test_data.csv",index=False)


# In[ ]:




