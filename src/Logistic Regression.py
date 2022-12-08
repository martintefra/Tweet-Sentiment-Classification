#!/usr/bin/env python
# coding: utf-8

#  # Load positive tweets & remove duplicate

# In[2]:


train_pos_class = []


# In[3]:


filename1 = 'data/train_pos.txt'
with open(filename1) as f1:
    train_pos = f1.readlines()
len(train_pos)


# In[4]:


#Remove duplciate positive tweets
train_pos = list(dict.fromkeys(train_pos))
len(train_pos)


# In[5]:


#Put each positive tweet in lower case in its own list with correct label (1)
for line in train_pos:
    train_pos_class.append([line.lower(),1])


# # Load negative tweets & remove duplicate
# 

# In[6]:


train_neg_class = []


# In[7]:


filename2 = 'data/train_neg.txt'
with open(filename2, encoding="utf8") as f2:
    train_neg = f2.readlines()
len(train_neg)


# In[8]:


#Remove duplciate negative tweets
train_neg = list(dict.fromkeys(train_neg))
len(train_neg)


# In[9]:


#Put each negative tweet in lower case in its own list with correct label (0)
for line in train_neg:
    train_neg_class.append([line.lower(),0])


# # Cleaning Data
# 

# In[10]:


from preprocessing import*


# In[11]:


#Concatenate the two training sets of positive and negative tweets
for line in train_neg_class:
    train_pos_class.append(line)


# In[12]:


train_data = train_pos_class


# In[13]:


train_df = pd.DataFrame(train_data)

#Labeling
train_df.columns = ["text", "cat_label"]


# In[14]:


train_df = clean_data(train_df)


# In[15]:


train_df.head(2)


# In[16]:


train_df.tail(2)


# # Test Data

# In[17]:


filename3 = 'data/test_data.txt'
with open(filename3) as f3:
    test_data = f3.readlines()
len(test_data)


# In[18]:


#Remove duplciate negative tweets
test_data = list(dict.fromkeys(test_data))
len(test_data)


# In[19]:


test_df = pd.DataFrame(test_data)
test_df.columns = ['text']
test_df


# In[20]:


test_df = clean_data(test_df)


# # Logistic Regression

# In[21]:


# Import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Create a CountVectorizer to convert the text into numerical data
vectorizer = CountVectorizer()

# Fit the vectorizer on the training data and transform it into numerical form
X_train = vectorizer.fit_transform(train_df["text"])

# Extract the labels from the training data
y_train = train_df["cat_label"]

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)


# # Predictions

# In[22]:


# Transform the test data into numerical form using the fitted vectorizer
X_test = vectorizer.transform(test_df["text"])

# Use the trained model to make predictions on the test data
predictions_log = model.predict(X_test)


# In[23]:


for i in range(len(predictions_log)):
    if(predictions_log[i] == 0):
        predictions_log[i] = -1
        
predictions_log


# In[25]:


predictions_log_df = pd.DataFrame({'Id': range(1, len(predictions_log) + 1), 'Prediction': predictions_log})

predictions_log_df.to_csv("predictions_log.csv",index=False)

