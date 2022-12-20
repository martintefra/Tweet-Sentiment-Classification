#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


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


# In[5]:


def get_most_common_words(sentence):
    words = sentence.split()

    # Create a dictionary to hold the count of each word
    word_count = {}

    # Iterate over the list of words and increment the count for each word
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Sort the dictionary by the count of each word in descending order
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # Output a list of tuples with the words and their counts
    word_count_list = [(word, count) for (word, count) in sorted_word_count]

    return word_count_list


# In[3]:


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


# In[4]:


def main(df_train,df_test): 
    whole_neg = " ".join(train_neg_class.text.values)
    word_neg_rep = get_most_common_words(whole_neg)
    whole_pos = " ".join(train_pos_class.text.values)
    word_pos_rep = get_most_common_words(whole_pos)
    
    sum_pos = 0
    sum_neg = 0
    
    for (word, apparition) in word_pos_rep:
        sum_pos = sum_pos + apparition

    for (word, apparition) in word_neg_rep:
        sum_neg = sum_neg + apparition
        
    df_test.columns = ['text']
    
    y_pred = []
    
    for i in range(len(df_test)):
        print(i)
        if (posteriori_probability(test_df['text'][i]) >= 0.5):
            y_pred.append(1)
        else:
            y_pred.append(-1) 
            
    test_data_df = pd.DataFrame(y_pred)
    test_data_df['Id'] = range(1, len(test_data_df) + 1)
    columns_titles = ["Prediction","Id"]
    test_data_df=test_data_df.reindex(columns=columns_titles)
    test_data_df.columns = ["Id", "Prediction"]
    
    return test_data_df


# In[ ]:




