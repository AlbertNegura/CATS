# CATS
Creative Automatic Title from Synopsis generator

Authors: Jonas Molz, Albert Negura

![Alt text](CATS.jpg?raw=true "Title")

This is a project done as part of the (2018) Natural Language Processing Course at the University of Maastricht.

## DESCRIPTION
This system represents an automatic title generator which attempts to utilize various NLP techniques to create a new title for a movie based on the Synopsis / genres provided.

To do so, we first look at existing movie synopses and attempt to do some genre-based classifcation. Using this, a mapping for each word in the title to each genre in generated, where the frequency of the word appearing in the respective genres is used as a way to weigh the word.

This weight is then intended to be passed as an input to an RNN (LSTM) as described in [Konstantin Lopryrev's paper](https://arxiv.org/pdf/1512.01712.pdf). This RNN takes an input text (summary, synopsis, etc.) and encodes it into a distributed representation. A decoder is then used to transform the representation into a title.

The system currently follows the flowchart on the left, and is intended to follow the flowchart on the right:

![Alt text](Process.png?raw=true "Title")

## DATASETS
Datasets used are 

Wikipedia: https://www.kaggle.com/jrobischon/wikipedia-movie-plots

MPST: https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-using-tags

The summaries on the datasets contain a variety of words, with names being the most prominent for every movie. Furthermore, some words (adverbs or articles such as 'the') have a much higher frequency than most other words (a few hundred fold increase in most cases).

![Alt text](WordDistribution.png?raw=true "Title")

## NOTEBOOKS
The following notebooks are available in this repo:

1. CATS-MAIN: contains all the preprocessing steps that have been / can be done with the data. Will (hopefully) contain the generator.

2. GenrePredictor: predicts the genre based on the synopsis using a simple Multinomial Naive Bayes model.

## MODELS

## DOWNLOAD
All of the models / datasets / embeddings (+ figures + notebooks) can be downloaded from this link:
https://drive.google.com/drive/folders/1Y_gj1PWEtBZbNSoc2VpmRhNXSalgD8sy?usp=sharing

## RESULTS
A few RAM issues on training the model used.

![Alt text](chrome_2019-05-27_19-44-12.png?raw=true "Title")
