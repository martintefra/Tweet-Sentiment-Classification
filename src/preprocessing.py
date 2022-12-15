#Basic Libraries
import re
import string
import numpy as np 
import pandas as pd

#Data Processing Libraries
import spacy
from nltk.stem import WordNetLemmatizer

#import the stopword list from the spacy library 
sp = spacy.load('en_core_web_sm')
#Counter appliation in order to improve the running 
spacy_stopwords = list(sp.Defaults.stop_words)
stop_words = spacy_stopwords + ['im', "i'm", 'dont','dunno', 'cant',"'s", 'u', 'x','user','url','rt','lol', '<user>', '<url>', '..','...']
lemmatizer = WordNetLemmatizer()

#cleaning "pipeline"  
def clean_data(text, stopwords, lemmatization):

      #perform casefolding
      text = text.casefold()
      #remove different tags for instance "<user>,<url>" for each twitter
      text = re.sub('<[^<]+?>','', text)
      #remove digits for each twitter
      text = ' '.join(text_ for text_ in text.split() if not text_.isdigit())
      #remove punctuations for each twitter
      text = ' '.join(text_ for text_ in text.split() if text_ not in string.punctuation)
      #remove the tokens length less than 2 
      text = ' '.join(text_ for text_ in text.split() if len(text)>2)
      
      if lemmatization :
          #perform lemmatization
          text = ' '.join(lemmatizer.lemmatize(text_)  for text_ in text.split() )
      
      if stopwords:
          #remove the stopwords
          text = ' '.join([word for word in text.split() if word not in stop_words])
      return text


#Load the data and run the preprocessor pipeline 
class Preprocessor:
    def __init__(self):
        """Init function
        """
    def load_data(preprocessed=True, directory="data"):
        POS_DATASET = directory + "/train_pos.txt"
        NEG_DATASET = directory + "/train_neg.txt"

        #import the data
        pos_data = pd.read_fwf(POS_DATASET, header=None, names=["tweets"])
        pos_data["labels"] = 1
        neg_data = pd.read_fwf(NEG_DATASET, header=None, names=["tweets"])
        neg_data["labels"] = 0
        data = pd.concat([pos_data, neg_data], ignore_index=True)
        np.random.seed(500)
        #shuffle the merge data
        data = data.iloc[np.random.permutation(len(data))]
        #remove the nan rows
        data.dropna(subset = ["tweets"], inplace=True)
        #remove the duplicates
        data.drop_duplicates(subset = "tweets", keep = False, inplace = True)
        #clean the data
        data['tweets'] = data['tweets'].apply(lambda x : clean_data(x, stopwords=True,lemmatization=True))
        
        #remove empty lines if any  
        data.dropna(subset = ["tweets"], inplace=True)

        X = data['tweets'].values
        y = data['labels'].values

        return np.array(X), np.array(y)