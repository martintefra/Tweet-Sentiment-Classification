#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import *
from tokenizers import *
import os
import json
import pandas as pd
import numpy as np


# In[ ]:


from simpletransformers.classification import ClassificationModel

model = ClassificationModel("roberta", "roberta-base",use_cude = False)


# In[ ]:


def main(df_train,df_test):
    model.train_model(df_train)
    df_test.columns = ['text']
    
    string = []
    for i in range(len(test_df)):
        string.append(df_test['text'][i])
        
    predictions, raw_outputs = model.predict(string)
    
    for i in range(len(predictions)):
    if(predictions[i] == 0):
        predictions[i] = -1
        
    test_data_df = pd.DataFrame(predictions)
    test_data_df['Id'] = range(1, len(test_data_df) + 1)
    test_data_df.columns = ["Id", "Prediction"]
    columns_titles = ["Prediction","Id"]
    test_data_df=test_data_df.reindex(columns=columns_titles)
    test_data_df.columns = ["Id", "Prediction"]
    
    return test_data_df


# In[ ]:




