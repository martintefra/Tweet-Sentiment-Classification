# Sentiment Analysis of Tweets

## About the project

This project aims to build a model to accurately predict the sentiment of tweets. Naive Bayes Classifier, Logistic Regression, RoBERTa, and a combination of multiple LSTM models are the four techniques we apply for the project.


## Overview

LSTM_models.py (LSTM Neural Networks Models) and run.ipynb the final notebook that generates the submissions, are the primary script/notebook for this project. We explored a number of classification techniques, including Naive Bayes, Logistic Regression, and RoBERTa (for the short datasets owing to a lack of resources), however the combination of LSTM neural networks produced the best accuracy for us. For the purpose of completeness, the other ML implementations are provided in this repository; read the README for additional information.

LSTM_models.py : is a script that trains the model using the preprocessed training dataset and generates predictions after preprocessing.py has been executed.
run.ipynb: The final notebook contains the preprocessing, implementation, and fitting phases for each model, as well as the creation of the submission file that we post to aicrowd.

All other and helpers scripts are stored in sc/ directory.

## Dependencies
To run our code from scripts or even σιμπλερ by running the run.ipynb file you will need to install the following libraries:

- wordninja
- emot
- contractions
- spacy
- nltk
- simpletransformers
- transformers
- tokenizers

The scripts and pynotebook assume that the train pos full and train neg full, as well as the 1.9GB of pretrained Glove embeddings as glove.twitter.27B.200d.txt, which you can download by the following commands,
$ wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -O glove-stanford/glove.twitter.27B.zip
$ unzip glove-stanford/glove.twitter.27B.zip -d glove-stanford/
are under the direction of data because we were able to import the full dataset to github. 

## Data

The data for this project consists of two sets of 1,250,000 tweets, one with positive emotions and one with negative emotions.

## Model

We have implemented four different models for this project:

- Naive Bayes Classifier
- Logistic Regression
- RoBERTa
- LSTM

## Evaluation

We have evaluated our models using the Accuracy and F1 score metrics.

## Usage

To use the trained model to classify new tweets, follow the instructions in the run.ipynb file.
