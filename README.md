# Sentiment Analysis of Tweets

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

As a baseline, we here provide sample code using word embeddings to build a text classifier system.

Submission system environment setup:

1. The dataset is available from the AIcrowd page, as linked in the PDF project description

 Download the provided datasets `twitter-datasets.zip`.

2. To submit your solution to the online evaluation system, we require you to prepare a “.csv” file of the same structure as sampleSubmission.csv (the order of the predictions does not matter, but make sure the tweet ids and predictions match). Your submission is evaluated according to the classification error (number of misclassified tweets) of your predictions.

*Working with Twitter data:* We provide a large set of training tweets, one tweet per line. All tweets in the file train pos.txt (and the train pos full.txt counterpart) used to have positive smileys, those of train neg.txt used to have a negative smiley. Additionally, the file test data.txt contains 10’000 tweets without any labels, each line numbered by the tweet-id.

Your task is to predict the labels of these tweets, and upload the predictions to AIcrowd. Your submission file for the 10’000 tweets must be of the form `<tweet-id>`, `<prediction>`, see `sampleSubmission.csv`.

Note that all tweets have already been pre-processed so that all words (tokens) are separated by a single whitespace. Also, the smileys (labels) have been removed.

## Overview

This project aims to build a model to accurately predict the emotions of tweets. We have implemented four different models: Naive Bayes Classifier, Logistic Regression, RoBERTa, and LSTM.:

## Getting Started
 

To run our code, open the run.ipynb file and select the model you want to test. You will need to install the following libraries:

- wordninja
- emot
- contractions
- spacy
- nltk
- simpletransformers
- transformers
- tokenizers

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