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

![Alt text](Pics/Break_Up_WordCloud.png?raw=true "Title")

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
Genre Classification can be overfit very easily, mostly due to class imbalance. Tag generation suffers from a similar effect, although partly because the network has been undertrained due to memory issues.

Handpicked Genres:

| Movie Title   | Actual Genres | Generated Genres | Actual Tags | Generated Tags |
| ------------- |:-------------:|:-----------------|:------------|:---------------|
| Deadpool | Thriller | Sports/Comedy, Horror Thriller, Action/Fantasy | Violence, Humor | Murder, Violence, Revenge |
| Scarface | Crime, Drama | Anthology Film, Horror Thriller, Action/Fantasy | Cruelty, Murder, Dramatic, Cult, Violence, Atmospheric, Action, Romantic, Revenge, Sadist| Murder, Violence, Revenge |
| The Mummy (1999) | Action | Sports/Comedy, Horror Thriller, Action/Fantasy | Revenge | Murder, Violence, Revenge |
| Iron Man 2 | Superhero | Sports/Comedy, Horror Thriller, Action/Fantasy | Good versus Evil, Violence | Murder, Violence, Revenge |
| The Godfather II | Crime/Drama | Sports/Comedy, Horror Thriller, Action/Fantasy | Violence, Humor, Murder | Murder, Violence, Revenge |
| Les Misérables | Period Drama | Sports/Comedy, Horror Thriller, Action/Fantasy | Satire | Murder, Violence, Revenge |
| We're the Millers | Comedy | Anthology Film, Horror Thriller, Action/Fantasy | Humor | Murder, Violence, Revenge |
| The Texas Chainsaw Massacre | Horror | Anthology Film, Horror Thriller, Action/Fantasy | Revenge, Murder, Violence, Flashback | Murder, Violence, Revenge |

### Clearly the tag-generation has a problem with humanity...
![Alt text](Pics/network.png?raw=true "Avoid making it self-aware!")

Example Output:

```python
Best prediction for Genres: 

['sports/comedy', 'horror thriller', 'action, fantasy']

Actual Genres: 

['thriller']

Title: 

['Deadpool']
```

Note that the generated titles do not contain common English adverbs and articles (the, for, of, just, etc.).

Handpicked Titles:

| Movie Title   | Synopsis Preview | Generated Titles  |
| ------------- |:-------------:|:----- |
| 28 Weeks Later | During the original outbreak of the Rage Virus, Don, his wife Alice and four more survivors hide in a barricaded cottage on the outskirts of London. They hear a terrified boy... | Happy Country |
| The Starving Games | Kantmiss Evershot practices archery in the forest, but her boyfriend, Dale, surprises her; the arrow accidentally hits the Wizard of Oz. They return to District 12, where... | Sequel: Macho Dawn |
| The Matrix | Computer programmer Thomas Anderson, living a double life as the hacker "Neo", feels something is wrong with the world and is puzzled by repeated online encounters with the cryptic phrase... | Private Death Man |
| Predators | Royce awakens to find himself parachuting into an unfamiliar jungle. He meets several others who arrived in the same manner: Mexican drug cartel enforcer Cuchillo, Spetsnaz... | Roommates |
| Amadeus | An elderly Antonio Salieri confesses to the murder of his former colleague, Wolfgang Amadeus Mozart, and attempts to kill himself by slitting his throat. Two servants take him to a mental... | Lucky Woman Oprah |
| Phil the Alien | Phil is an extraterrestrial with shape-shifting ability and telekinetic powers. After crash-landing in Northern Ontario, Phil befriends a red neck child, his father, and... | (The) Story |
| The Secret Life of Happy People | Thomas Dufresne (Paquet) is the black sheep of his bourgeois family. One day, he meets a free-spirited waitress named Audrey (Catherine Deléan) who changes his life... | Rose Life |
| Non-Stop | Two U.S. Air Marshals, Jack Hammond and alcoholic Bill Marks, separately board a British Aqualantic Airlines Boeing 767 from New York City to London. Marks sits next to Jen... | Island Love |
| Two Men Went to War | Sergeant Peter King and Private Leslie Cuthbertson of the Royal Army Dental Corps passionately desire to see active service, but are held back. Armed with two revolvers... | Doraemon: White Trouble Love |
| Robin and Marian | An aging Robin Hood is a trusted captain fighting for King Richard the Lionheart in France, the Crusades long over. Richard orders him to take a castle that is rumoured to... | Big Angel Valley Takes Breaking |
| From Hell it Came | A South Seas island prince is wrongly convicted of murder and executed by having a knife driven into his heart, the result of a plot by a witch doctor (the true murderer)... | Nobody Happy |
| Big Night | On the New Jersey Shore in the 1950s, two Italian immigrant brothers from Calabria own and operate a restaurant called "Paradise." One brother, Primo, is a brilliant, perfectionist chef who chafes... | Town Daddy / May Bunny |
| Kama Sutra: A Tale of Love | Set in 16th century India, this movie depicts the story of two girls who were raised together, though they came from different social classes... | Mr Goodbye Wife Star / Last Debutante / Defiance Valley |
| This World, Then the Fireworks | As children, Marty and Carol Lakewood, fraternal twins, witness a brutal murder involving their father. They grow up to become depraved and incestuous adults... | Play Princess |
| Endless Love | In suburban Chicago, teenagers Jade Butterfield and David Axelrod fall in love after they are introduced by Jade\'s brother Keith. Jade\'s family is known...| Dangerous Man/Underworld |
| Carrie | Alone in her home, Margaret White, a disturbed religious fanatic, gives birth to a baby girl, intending to kill the infant but changes her mind. Years later, her daughter Carrie, a shy... | Adventure: Woman Menstruation |
| Agnes of God | In a Roman Catholic convent near Montreal, Quebec, Canada, during evening prayers, the nuns hear screams coming from ther oom of Sister Agnes, a young novice.Agnes is found in her room bleeding profusely...     |  Johnny Boy/Holiday |
| Nowhere to Run | In rural Texas, 1960 an age of good times and innocence, when growing up was supposed to be easy six high school seniors know the terrible secret...      | Keeps Zoo Enchanted/Walk Two |
| Tarantula | A severely deformed man stumbles through the Arizona desert, falls and dies. Dr. Matt Hastings, a doctor in the nearby mall... | Cross Man |
| Care Bears: Journey to Joke-a-lot | The Care Bears live in a cloud-filled land known as Care-a-lot ("With All Your Heart"). One of the Bears, Grumpy, is working on a rainbow carousel for the upcoming... | Dick Sands Master: East Island Bidel |
| Little Shop of Horrors | A three-girl "Greek chorus" Crystal, Ronnette, and Chiffon introduce the movie, warning the audience that some horror is coming their way... | Farb/Stoolie | 
| Frankenstein | In a European village, a young scientist, named Henry Frankenstein, and his assistant Fritz, a hunchback, piece together a human body, the parts of which have been collected... | Cry |
| Just for Fun | When English teenagers win the right to vote, the established political parties compete for their support. However, when the Prime Minister cuts... | Old Time/Dim Moon/Shrike/Hot |
| Voice in the Wind | Jan Foley (Lederer), an amnesiac Czech pianist, is a victim of Nazi torture for playing a banned song. Living under a new identity on the island of French-governed Guadalupe, Jan tries to... | Incredible Granach/Frankenstein |
| For Those in Peril | Aspiring RAF pilot Pilot Officer Rawlings (Ralph Michael) fails to make the grade in training and grudgingly accepts the alternative of joining the crew of Launch 183... | Mundanity/Love Love |
| Now and Forever | A lazy and irresponsible Jerry Day (Gary Cooper), desperate for quick cash, is willing to sell the custody rights of his 6-year-old daughter Penelope, nicknamed Penny... | Wings Green/Silver Secret/Submarine |
| The Las Vegas Story |Happy (Hoagy Carmichael), as the piano player at the Last Chance casino in Las Vegas, wonders what split up Linda Rollins (Jane Russell) and Dave Andrews (Victor Mature)... | Go Black |


Example Output of Title Generation Network:

```python
(Actual Title) HEAD: gamera vs gyaos
(Processed Synopsis) DESC: series volcanoes^ erupt japan eruption mt futago^ shizuoka^ prefecture attracts gamera whose arrival witnessed young boy named eiichi gamera climbs volcano research team dispatched volcano find gamera study effects eruption meanwhile chuo^ expressway corporation building roadway^ nearby local villagers refuse leave research team helicopter destroyed sonic beam emitted^ cave mountains reporters informed bodies found culprit gamera volcanic eruption announcement made soon one reporters okabe leaves site roadcrew^ foreman shiro tsutsumi arrive protest area simultaneously okabe
(Generated Titles, higher is better) HEADS:
8.712603449821472 midnight
8.734845966100693 serena
10.19324865937233 night annihilates^
14.785994216799736 cat dream
15.782405376434326 happy jackson
```

'Mannix' is often generated as the title of the movie. A preliminary analysis reveals that 'Mannix' is the most dissimilar word to all others in the corpus, so the network decides to choose it when it cannot "think" of a good alternative.

## DOWNLOAD
All of the models / datasets / embeddings (+ figures + notebooks) can be downloaded from this link:
https://drive.google.com/drive/folders/1Y_gj1PWEtBZbNSoc2VpmRhNXSalgD8sy?usp=sharing

A few RAM issues on training the model used.

