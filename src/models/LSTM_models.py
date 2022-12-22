
# 6 LSTM models + XGboost Classifier

#Basic Libraries
import pandas as pd
import numpy as np 

from sklearn.metrics import f1_score
import sklearn.metrics as metrics

#xgboost
from xgboost import XGBClassifier

#Build the LSTM model
import tensorflow as tf
import pickle as cPickle

from tensorflow import keras
from keras.preprocessing.text import  Tokenizer
from keras.utils import pad_sequences


from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.layers import Embedding, SpatialDropout1D , Conv1D
from keras.layers import Bidirectional, LSTM, Dense, Dropout,Masking,Activation
from keras.optimizers import RMSprop

import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # SGD, RMSprop




def embedding_index_Glove():

    #retrieve the pretrained embeddings and store them as a dictionary
    embeddings_index = {}
    f = open('../data/glove.twitter.27B.200d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
    
def main(df_train,df_test,embeddings_index):
    
    X_test = Preprocessor.load_data(preprocessed=True,Train_data=False)
    X_full = Preprocessor.load_data(preprocessed=True,Train_data=True)
    X_full_preprocessed
    y_train=df_train.labels


    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(df_train.tweets)
    X_train = tokenizer.texts_to_sequences(df_train.tweets)#convert each word to a integer based on the tokenizer
    
    vocab_size=len(tokenizer.word_index)+1 
    max_len=40  # maxh length of each tweet is set to 40 words s(so will be performed padding and truncation )
    X_train = pad_sequences(X_train, padding='post'  ,maxlen=max_len)
    
    
    X_test = tokenizer.texts_to_sequences(df_test.tweets)#covert each word to a integer based on the tokenizer
    X_test = pad_sequences(X_test, padding='post'  ,maxlen=max_len)

    #form our embedding matrix for each word that appears in our dataset based on pretrained glove embeddings

    embeddings_index=embedding_index_Glove()
    embedding_matrix = np.zeros((vocab_size , 200))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    #Train the models

    EPOCHS=6
    BATCH_SIZE=1024
    embedding_size=200
    num_of_words=40


    #MODEL 1 

    model1 = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len , trainable=False,mask_zero=True) 
    model1.add(embedding_layer)
    model1.add(Masking(mask_value=0.0)) 
    model1.add(Bidirectional(LSTM(512)))
    model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(1))
    model1.add(Activation('sigmoid'))

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model1.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)


    #model1.save('/content/drive/MyDrive/model_1',save_format="h5")


    #MODEL 2

    model2 = Sequential()
    embedding_layer2 = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len , trainable=False) 
    model2.add(embedding_layer2)
    model2.add(LSTM(100))
    model2.add(Dense(64))
    model2.add(Dense(1, activation='sigmoid'))

    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model2.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

    #model2.save('/model_2',save_format="h5")


    #MODEL 3

    model3 = Sequential()
    embedding_layer3 = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len , trainable=False) 
    model3.add(embedding_layer3)
    model3.add(LSTM(1024))
    model3.add(Dropout(0.4))
    model3.add(Dense(512, activation='relu'))
    model3.add(Dropout(0.4))
    model3.add(Dense(512,activation='relu'))
    model3.add(Dropout(0.4))
    model3.add(Dense(512,activation='relu'))
    model3.add(Dense(1, activation='sigmoid'))

    model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model3.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

    #model3.save('/content/drive/MyDrive/model_3',save_format="h5")


    #MODEL 4

    model4 = Sequential()
    embedding_layer2 = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len , trainable=False,mask_zero=True) 
    model4.add(embedding_layer2)
    model4.add(Masking(mask_value=0.0))
    model4.add(LSTM(512,return_sequences=True))
    model4.add(Dropout(0.3))
    model4.add(LSTM(512,return_sequences=True))
    model4.add(LSTM(265))
    model4.add(Dense(64, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(Dense(1))
    model4.add(Activation('sigmoid'))

    model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model4.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

    #model4.save('/content/drive/MyDrive/model_4',save_format="h5")

    #MODEL 5

    model5 = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=num_of_words , trainable=False,mask_zero=True) 
    model5.add(embedding_layer)
    model5.add(Masking(mask_value=0.0)) 
    model5.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model5.add(MaxPooling1D(pool_size=2))
    model5.add(LSTM(256))
    model5.add(Dense(64, activation='relu'))
    model5.add(Dropout(0.5))
    model5.add(Dense(1))
    model5.add(Activation('sigmoid'))

    model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model5.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

    #MODEL 6 

    model6 = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=max_len , trainable=False) 
    model6.add(embedding_layer)
    model6.add(Bidirectional(LSTM(1024)))
    model6.add(Dense(512, activation='relu'))
    model6.add(Dropout(0.4))
    model6.add(Dense(512, activation='relu'))
    model6.add(Dropout(0.4))
    model6.add(Dense(512, activation='relu'))
    model6.add(Dropout(0.4))
    model6.add(Dense(1))
    model6.add(Activation('sigmoid'))

    model6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model6.fit(X_train,y_train , batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)
    #model6.save('/content/drive/MyDrive/model_6',save_format="h5")


    #retrive the training predictions and testing predictions in order to combine these results using XGb classifier
    train1 = model1.predict(X_train, batch_size=128)
    test1 = model1.predict(X_test)

    train2 = model2.predict(X_train, batch_size=128)
    test2 = model2.predict(X_test)

    train3 = model3.predict(X_train, batch_size=128)
    test3 = model3.predict(X_test)

    train4 = model4.predict(X_train, batch_size=128)
    test4 = model4.predict(X_test)

    train5 = model5.predict(X_train, batch_size=128)
    test5 = model5.predict(X_test)

    train6 = model6.predict(X_train, batch_size=128)
    test6 = model6.predict(X_test)

    #combine all the training predictions and testing predictions in order to train a XBGbooster for more accuracy
    train = np.hstack((train1, train2, train3, train4, train5))
    test = np.hstack((test1, test2, test3, test4, test5))


    #train a XGB booster
    model = xgb.XGBClassifier().fit(train, y_train)
    y_pred = model.predict(test)
    y_pred=[-1 if y_p<0.5 else 1 for y_p in y_pred ]

    return y_pred
