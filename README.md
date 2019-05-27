# CATS
Creative Automatic Title from Synopsis generator

Authors: Jonas Molz, Albert Negura


This is a project done as part of the (2018) Natural Language Processing Course at the University of Maastricht.

## DESCRIPTION
This system represents an automatic title generator which attempts to utilize various NLP techniques to create a new title for a movie based on the Synopsis / genres provided.

To do so, we first look at existing movie synopses and attempt to do some genre-based classifcation. Using this, a mapping for each word in the title to each genre in generated, where the frequency of the word appearing in the respective genres is used as a way to weigh the word.

This weight is then utilized when automatically selecting a random number of words which are then used to generate a title.

## DATASETS
Datasets used are 

Wikipedia: https://www.kaggle.com/jrobischon/wikipedia-movie-plots

MPST: https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-using-tags

The summaries on the datasets contain a variety of words, with names being the most prominant for every movie. Furthermore, some words (adverbs or articles such as 'the') have a much higher frequency than most other words (a few hundred fold increase in most cases).


## NOTEBOOKS
The following notebooks are available in this repo:

1. CATS-MAIN: contains all the preprocessing steps that have been / can be done with the data. Will (hopefully) contain the generator.
2. GenrePredictor: predicts the genre based on the synopsis using a simple Multinomial Naive Bayes model.

## MODELS

## RESULTS
A few RAM issues on training the model used.
![Alt text](chrome_2019-05-27_19-44-12.png?raw=true "Title")
