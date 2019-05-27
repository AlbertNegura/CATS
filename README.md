# CATS
Creative Automatic Title from Synopsis generator

## DESCRIPTION
This system represents an automatic title generator which attempts to utilize various NLP techniques to create a new title for a movie based on the Synopsis / genres provided.

To do so, we first look at existing movie synopses and attempt to do some genre-based classifcation. Using this, a mapping for each word in the title to each genre in generated, where the frequency of the word appearing in the respective genres is used as a way to weigh the word.

This weight is then utilized when automatically selecting a random number of words which are then used to generate a title.

## DATASETS
Datasets used are 
Wikipedia: https://www.kaggle.com/jrobischon/wikipedia-movie-plots
MPST: https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-using-tags
