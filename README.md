# CATS
Creative Automatic Title from Synopsis generator

Authors: Jonas Molz, Albert Negura

![Alt text](Pics/CATS.jpg?raw=true "Title")

This is a project done as part of the (2018) Natural Language Processing Course at the University of Maastricht.

## DESCRIPTION
This system represents an automatic title generator which attempts to utilize various NLP techniques to create a new title for a movie based on the Synopsis / genres provided.

To do so, we first look at existing movie synopses and attempt to do some genre-based classifcation. Using this, a mapping for each word in the title to each genre in generated, where the frequency of the word appearing in the respective genres is used as a way to weigh the word.

This weight is then intended to be passed as an input to an RNN (LSTM) as described in [Konstantin Lopryrev's paper](https://arxiv.org/pdf/1512.01712.pdf). This RNN takes an input text (summary, synopsis, etc.) and encodes it into a distributed representation. A decoder is then used to transform the representation into a title.

The system currently follows the flowchart on the left, and is intended to follow the flowchart on the right:

![Alt text](Pics/Process.png?raw=true "Title")

## DATASETS
Datasets used are 

Wikipedia: https://www.kaggle.com/jrobischon/wikipedia-movie-plots

MPST: https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-using-tags

The summaries on the datasets contain a variety of words, with names being the most prominent for every movie. 

![Alt text](Pics/breakup.png?raw=true "Title")

Furthermore, some words (adverbs or articles such as 'the') have a much higher frequency than most other words (a few hundred fold increase in most cases).

![Alt text](Pics/WordDistribution.png?raw=true "Title")

The average length of movie titles in the dataset is about 2 words, while the average length of the Wikipedia summaries is 350 words.

![Alt text](Pics/WordLengthOfTitles.png?raw=true "Title")
![Alt text](Pics/WordLengthOfSummaries.png?raw=true "Title")


## NOTEBOOKS
The following notebooks are available in this repo:

1. [CATS-MAIN.ipynb](https://github.com/FireLionX/CATS/blob/master/CATS-Main.ipynb): contains all the preprocessing steps that have been / can be done with the data for this project. Furthermore, contains the Naive Bayes genre classifier.

2. [GenrePredictor.ipynb](https://github.com/FireLionX/CATS/blob/master/GenrePredictor.ipynb): contains a Python 3 implementation of the Lopyrev model, which attempts to generate a title based on the synopsis. Was supposed to contain a genre-based approach to generating a headline.

3. [vocabulary-embedding.ipynb](https://github.com/udibr/headlines): Python2: generates pickle dumps containing processed training data and initialized embedding.

4. [train.ipynb](https://github.com/udibr/headlines): Python2: trains a network for Lopyrev's model, generating titles from the synopsis of movies, and saves the model.

5. [predict.ipynb](https://github.com/udibr/headlines): Python2: generates predictions based on the saved Lopyrev model.

6. [kar_et_al_tag_generator.ipynb](https://github.com/cryptexcode/folksonomication_source): generates tags based on Kar's model, which can then be used in conjunction with the genre classifier to generate headlines.

## MODELS
For Genre Classification, MultinomialNB was the primary model used. It has high accuracy on most genres, with decreasing accuracy depending on how common the genre is in the training dataset used.

![Alt text](Pics/Classifier.png?raw=true "Title")

For generating titles, Lopyrev's model was used and trained on 

![Alt text](Pics/TrainingLoss.png?raw=true "Title")

Usually, training the models requires a significant amount of RAM. This message appeared far too often minutes or, sometimes, hours into the training. A beefy machine is necessary for Lopyrev's or Kar et. al's model.

![Alt text](Pics/chrome_2019-05-27_19-44-12.png?raw=true "Title")

## EXAMPLES
Note that the generated titles do not contain common English adverbs and articles (the, for, of, just, etc.).
Handpicked Titles:

| Movie Title   | Synopsis Preview | Generated Titles  |
| ------------- |:-------------:|:----- |
| Endless Love | In suburban Chicago, teenagers Jade Butterfield and David Axelrod fall in love after they are introduced by Jade\'s brother Keith. Jade\'s family is known...| Dangerous Man/Underworld |
| Agnes of God | In a Roman Catholic convent near Montreal, Quebec, Canada, during evening prayers, the nuns hear screams coming from ther oom of Sister Agnes, a young novice.Agnes is found in her room bleeding profusely...     |  Johnny Boy/Holiday |
| Nowhere to Run | In rural Texas, 1960 an age of good times and innocence, when growing up was supposed to be easy six high school seniors know the terrible secret...      | Keeps Zoo Enchanted/Walk Two |
| Tarantula | A severely deformed man stumbles through the Arizona desert, falls and dies. Dr. Matt Hastings, a doctor in the nearby mall... | Cross Man |
| Little Shop of Horrors | A three-girl "Greek chorus" Crystal, Ronnette, and Chiffon introduce the movie, warning the audience that some horror is coming their way... | Farb/Stoolie | 
| Frankenstein | In a European village, a young scientist, named Henry Frankenstein, and his assistant Fritz, a hunchback, piece together a human body, the parts of which have been collected... | Cry |
| Just for Fun | When English teenagers win the right to vote, the established political parties compete for their support. However, when the Prime Minister cuts... | Old Time/Dim Moon/Shrike |
| Voice in the Wind | Jan Foley (Lederer), an amnesiac Czech pianist, is a victim of Nazi torture for playing a banned song. Living under a new identity on the island of French-governed Guadalupe, Jan tries to... | Incredible Granach/Frankenstein |
| For Those in Peril | Aspiring RAF pilot Pilot Officer Rawlings (Ralph Michael) fails to make the grade in training and grudgingly accepts the alternative of joining the crew of Launch 183... | Mundanity/Love Love |
| Now and Forever | A lazy and irresponsible Jerry Day (Gary Cooper), desperate for quick cash, is willing to sell the custody rights of his 6-year-old daughter Penelope, nicknamed Penny... | Wings Green/Silver Secret/Submarine |



Example Output of Title Generation Network:

(Actual Title) HEAD: gamera vs gyaos
(Processed Synopsis) DESC: series volcanoes^ erupt japan eruption mt futago^ shizuoka^ prefecture attracts gamera whose arrival witnessed young boy named eiichi gamera climbs volcano research team dispatched volcano find gamera study effects eruption meanwhile chuo^ expressway corporation building roadway^ nearby local villagers refuse leave research team helicopter destroyed sonic beam emitted^ cave mountains reporters informed bodies found culprit gamera volcanic eruption announcement made soon one reporters okabe leaves site roadcrew^ foreman shiro tsutsumi arrive protest area simultaneously okabe
(Generated Titles, higher is better) HEADS:
8.712603449821472 midnight
8.734845966100693 serena
10.19324865937233 night annihilates^
14.785994216799736 cat dream
15.782405376434326 happy jackson

'Mannix' is often generated as the title of the movie. A preliminary analysis reveals that 'Mannix' is the most dissimilar word to all others in the corpus, so the network decides to choose it when it cannot "think" of a good alternative.

Sometimes, the network has a sense of humour:

28 Weeks Later (horror movie) -> Happy Country

The Starving Games (parody movie) -> Sequel: Macho Dawn

## DOWNLOAD
All of the models / datasets / embeddings (+ figures + notebooks) can be downloaded from this link:
https://drive.google.com/drive/folders/1Y_gj1PWEtBZbNSoc2VpmRhNXSalgD8sy?usp=sharing

A few RAM issues on training the model used.

