#Basic Libraries
import re
import string
import numpy as np 
import pandas as pd

from collections import Counter

#read html files 
import requests

#Data Processing Libraries
import spacy
from nltk.stem import WordNetLemmatizer

#Text processing libraries
#for expansion the contractions
import contractions 
#import emoticons in order to replace them with appropriate words
from emot.emo_unicode import EMOTICONS_EMO
#import wordninja in order to split the words
import wordninja

DATA_PATH = "../data"

#import the stopword list from the spacy library 
sp = spacy.load('en_core_web_sm')
#Counter appliation in order to improve the running 
spacy_stopwords = list(sp.Defaults.stop_words)
stopwords_dict = Counter(spacy_stopwords)
lemmatizer = WordNetLemmatizer()

#we use these two dataset in ADA homeworks
url_positive = "https://ptrckprry.com/course/ssd/data/positive-words.txt"
rsp = requests.get(url_positive)
lines = rsp.text.strip("\n").split("\n")
positive_words = lines[lines.index('a+'):]

url_negative = "https://ptrckprry.com/course/ssd/data/negative-words.txt"
rsp = requests.get(url_negative)
lines = rsp.text.strip("\n").split("\n")
negative_words = lines[lines.index('2-faced'):]

#A more robust preprocessing phase 
#import the stopword list from the spacy library 
#Counter appliation in order to improve the running 
EMOTICONS_EMO[':d'] = 'laughing'
EMOTICONS_EMO['<3'] = 'red heart'

def expansion_patterns(text):
    """Expand the contractions
    For instance I'll or I've been to I will and I have been.
    
    Output:
        text: the text after the expansion
    """
    expansion_patterns = [(' nd ',' and '),(' wa ',' was '),(' donnow ',' do not know '),(' i\'ts ','it is '),
                      (' dem ',' them '),(' #+ha+ha ',' haha '),(' i\'ts ','it is '),(' i\'ts ','it is '),(' n+a+h+ ', ' no '),
                      (' n+a+ ', ' no '),(' w+o+w+', 'wow '),('y+a+y+', 'yay'),('y+[e,a]+s+', 'yes'),
                      (' ya ', ' you '),('n+o+', 'no'),('a+h+','ah'),('muah','kiss'),(' y+u+p+ ', ' yes '),(' y+e+p+ ', ' yes '),
                      (' ima ', ' i am going to '),(' woah ', ' wow '),(' wo ', ' wow '),(' aw ', ' cute '), 
                      (' lmao ', ' haha '),(' lol ', ' haha ')]
    patterns = [(re.compile(regex_exp, re.IGNORECASE), replacement) for (regex_exp, replacement) in expansion_patterns]
    for (pattern, replacement) in patterns:
        (text, _) = re.subn(pattern, replacement, text)
    return text

LEN_DATA = 0
COUNT = 0

def increment():
    global COUNT
    COUNT = COUNT+1


#TODO: get synonyms for each word (PyDictionary) and replace it with the most frequent one (wordfreq)
def clean_data(text, stopwords=True, lemmatization=True):
    """Preprocessing phase
    Input:
        text: the text to be cleaned
        stopwords: if True remove the stopwords
        lemmatization: if True apply the lemmatization
    Output:
        text: the text after the preprocessing phase
    """

    """Emojis to words
    For instance if we have 'text text <3 :d text :D' will be 'text text red heart positive laughing positive text Laughing'
    """
    text = ' '.join(EMOTICONS_EMO.get(word) if word in EMOTICONS_EMO.keys() else word for word in text.split()) 

    """Perform casefolding"""
    text = text.casefold()

    """Remove ponctuation"""
    text = ' '.join(text_ for text_ in text.split() if text_ not in string.punctuation)

    """Remove numbers
    It remove all numbers not just digits since it doesn't give so much information for the purpose of sentimental analysis
    For instance '#5words 625' with be '#words and then will be "words" after removing the hashtags in the later phase of preprocessing
    """
    text = ' '.join(re.sub('(\d+(\.\d+)?)','',word) if re.search('(\d+(\.\d+)?)',word) else word for word in text.split() ).strip()

    """Remove tags
    It remove different tags for instance "<user>,<url>" for each twitter.
    """
    text = re.sub('<[^<]+?>','', text)


    """Remove multiply commas and dots everywhere in tweets.
    For instance '....' will be '.' and ',,,,,' will be ','
    """
    text = re.sub('\.|,*','', text)

    """Expand the contractions
    For instance I'll or I've been to I will and I have been.
    """
    text = expansion_patterns(text)
    text = ' '.join(contractions.fix(text_) for text_ in text.split() ) 

    """Remove the stopwords"""
    if stopwords:
        text = ' '.join([word for word in text.split() if word not in stopwords_dict])         
    
    """Split words within hashtags
    For instance the hashtags #happythoughts and #cryyoureffingeyesout 
    will be respectively 'happy thoughts' and 'cry your effing eyes out'
    """
    text=' '.join( ' '.join(wordninja.split(word_[1:])) if word_.startswith('#') else word_ for word_ in text.split())

    """Lemmatization"""
    if lemmatization :
        text = ' '.join(lemmatizer.lemmatize(text_)  for text_ in text.split() )

    """Word replacement using predefined dictionaries
    It uses the positive and negative sentimental analysis 
    to add the 'positive' and 'negative' token after the word.
    For instance 'happy' will be 'happy positive' and 'sad' will be 'sad negative'
    """
    text = ' '.join(word+" positive" if word in positive_words else word for word in text.split() )
    text = ' '.join(word+" negative" if word in negative_words else word for word in text.split() )

    """Remove the tokens length less than 2 again 
    if some may appear after the above preprocessing.
    """
    text = ' '.join(text_ for text_ in text.split() if len(text_)>2)

    increment()
    if(COUNT%10000==0):
        print("Cleaning data: ",COUNT, " tweets cleaned")
    return  text.strip()


#Load the data and run the preprocessor pipeline 
class Preprocessor:
    def __init__(self):
        """Init function
        """
    def load_data(directory=DATA_PATH, test=False):
        """Load the data

        Output:
            data: pandas dataframe with the tweets and labels
        """
        #Load testing data
        if test:
            TEST_DATASET = directory + "/test_data.txt"

            #import the data
            with open(TEST_DATASET) as f:
                data = f.readlines()
                data = pd.DataFrame(data,columns=['tweets'])
                data.tweets=data.tweets.apply(lambda x :x[x.find(',')+1:])
                data["labels"] = None

        #Load training data
        else:
            POS_DATASET = directory + "/train_pos.txt"
            NEG_DATASET = directory + "/train_neg.txt"

            #import the data
            pos_data = pd.read_fwf(POS_DATASET, header=None, names=["tweets"])
            pos_data["labels"] = 1
            neg_data = pd.read_fwf(NEG_DATASET, header=None, names=["tweets"])
            neg_data["labels"] = 0
            data = pd.concat([pos_data, neg_data], ignore_index=True)

        global LEN_DATA 
        LEN_DATA = len(data)

        #shuffle the merge data
        np.random.seed(500)
        #shuffle the merge data
        data = data.iloc[np.random.permutation(len(data))]
        #remove the nan rows
        data.dropna(subset = ["tweets"], inplace=True)
        #remove the duplicates
        data.drop_duplicates(subset = "tweets", keep = False, inplace = True)
        #clean the data
        data['tweets'] = data['tweets'].apply(lambda x : clean_data(x, stopwords=True, lemmatization=True))
        #remove empty lines if any  
        data.dropna(subset = ["tweets"], inplace=True)

        return data