# Predicting Political Leanings from News Headlines

## Introduction:
* A lot of research has been done into media bias (Card et al 2015) and detecting the political leanings of an article based on its content (Smith et al 2010). In today's more partisan times, I was curious to know whether analysis of just a news article's headline would give you sufficient information to predict either the news source itself, or its political leanings. To my knowledge, not much research has been done into analysis of news headlines themselves (aside from Piotrkowicz et al. 2017), and whether or not the headline itself can predict a particular political leaning/the news source it is from. I'm interested in establishing a baseline for this task.
* To address this question, I considered six news sources: CNN, FoxNews, NPR, NBC, Breitbart, and Wall Street Journal (WSJ). These companies all publish news articles online. Three are known for leaning conservative, to varying extents: Fox News, WSJ, and Breitbart. The other three lean liberal. I downloaded the news headlines as published on Facebook, extracted features from them, and then ran a random forest model to establish a baseline accuracy.

## Data:
* I considered links shared on each news sources' Facebook timelines between July 2013 and July 2017. This gave me between 20,000 and 50,000 links for each news source.
* For each of these links, I used the 'link_name' attribute of the Facebook API results as the headline of the article. This is the title of the link the user would see in the preview box before clicking on the article, e.g. "Priebus out, Kelly in as White House chief of staff" in the following example.


## Features:
* I computed a number of features for each of these headlines:
    * Average sentence length, average word length, length of the headline in characters
    * Lexical richness, lexical density, Flesch readability score
    * Counts of each part of speech in the headline (nouns, adjectives, adverbs, prepositions, interjections, verbs, and pronouns). I used NLTK's POS tagger to do so.
    * Count of the number of uppercase words in the headline
    * Count of the number of superlatives in the headline (POS tagged JJS or RBS - adjective or adverb superlatives)
    * Sentiment scores (positive, negative, compound) using NLTK's Vader SentimentIntensity Analyzer for the headline overall, and the sum of the scores for each word in the headline
    * Top 10,000 multiword phrases in all of the headlines, using phrasemachine (O'Connor et al. 2016) - this seemed to have good support for social science terms, so I felt that this would be better for news headlines than using a standard named entity recognizer
    * Top 10,000 unigrams and bigrams from all the headlines
* Since these news headlines are all pulled from a social media website, I felt that news sources might use more clickbait-like headlines to attract readers, and that different news sources might do this to different extents, with different techniques. Therefore, I computed features based on what I read about detecting clickbait. 
    * Formality measure - this was used in Biyani et al (2016) and originally defined by Heylighen and Dewaele (1999). This measurement is based on the counts of different part-of-speech tags in the headline.
    * Words in the Downworthy list - these are words taken from github.com/snipe/downworthy that are frequently used in clickbait headlines. They were also used by Chakraborty et al (2016) to identify clickbait.
    * Number of hyperbolic words - words were considered hyperbolic if the Stanford CoreNLP sentiment value was greater than 4 or less than 1 (very positive or very negative); also used by Chakraborty et al (2016).
    * Starts with number, starts with adverb - two features that were used by Biyani et al (2016) that were ranked in their top 40 most important features for detecting clickbait

A number of these features showed meaningful trends. For example, we can see that NPR is significantly more lexically dense throughout the last four years, and that in general each news source has had a roughly consistent lexical density over time:

![Lexical density variation](/lexical_density.png)


Headline lengths (by number of characters) also seem to show significant differences. In the case of all but one news source (NBC), the lengths have increased over time. This suggests a time series analysis might be useful for future work.

![Headline length variation](/headline_length.png)


The formality measure also shows roughly steady differences between news sources.

![Formality measure variation](/formality_measure.png)


## Model:
I wanted to use roughly 10,000 headlines as datapoints, so I randomly sampled the data for 1,666 headlines per news source, and 9,996 headlines total. I computed the top 10,000 multiword phrases and top 10,000 unigrams and bigrams using these headlines. I combined the rest of the feature data for these 9,996 headlines, and then produced a label array with 6 classes, for the 6 newsgroups. I used 2,000 of these datapoints for my test set. I used a random forest model from scikit-learn, and did a grid search to find the optimal number of trees out of {40, 60, 80, 100, 120} trees. This, in both cases, used 120 trees in the best model.

## Results:
I first tried to predict the news source based on the feature data. Using the best model from the grid search, the model was able to learn the training set entirely, with an accuracy of 100%. However, the accuracy on the test set was only 49%. Here are the top 10 features and the normalized confusion matrix for the test set:
                        
![Top 10 features](/mif_news_sources.png)
![Confusion matrix](/cnf_matrix_news_sources.png)


From the confusion matrix, you see that NBC often gets mistaken for CNN, Breitbart gets mistaken for Fox, and Wall Street Journal gets mistaken for CNN.

When trying to predict the political leaning of the news source itself (CNN/NBC/NPR - liberal, Fox/Breitbart/WSJ - conservative), the model performed significantly better. It was also able to learn the training set entirely, and had an accuracy on the test set of 72%. Here are the top 10 features and the normalized confusion matrix for this test set:

![Top 10 features](/mif_party.png)
![Confusion matrix](/cnf_matrix_party.png)


Here, conservative news sources are often mistaken for liberal.

## Conclusions:
* Predicting the news source itself from the headline can be done with accuracy better than chance (49% accuracy as opposed to 17% by chance for 6 news sources). 
    * In incorrect predictions, NBC often gets mistaken for CNN, Breitbart gets mistaken for Fox, and Wall Street Journal gets mistaken for CNN.
* Predicting the political leaning from the headline can be done with reasonable accuracy (72%). 
    * In incorrect predictions, conservative news sources are often mistaken for liberal.
* Key features that allow you to make both predictions include the length of the headline, lexical density, number of nouns, a formality measure, the average word/sentence length, and the Flesch readability score of the headline.

## Next steps:
* Use Wikification to identify entities in the text - this was used for news headline text in Piotrkowicz et al. 2017
* Use OpinionFinder instead of NLTK for sentiment analysis; it's mentioned in multiple papers for sentiment analysis (O'Connor et al., etc.)
* Use Yano et al.'s method to identify “sticky bigrams” strongly associated with one party or another, rather than just using all unigrams and bigrams (Iyyer et al. 2014)
* Use LIWC to extract finer-grained emotions in the headlines and see if that increases model performance (used by Iyyer et al. 2014)

## More questions to ask:
* Can we predict the sentiment/other feature values of newer news headlines using a time series model, and how accurate would this be?
    * download and use July 2017 data as newer headlines/test set
* Are the changes in sentiment etc. between 2013 and now statistically significant?
* See if a neural net/nonlinear feature combinations would help the accuracy of both classifiers


## References:
Piotrkowicz et al. 2017. ["Automatic Extraction of News Values from Headline Text".](http://www.aclweb.org/anthology/E17-4007)
O'Connor et al. 2010. ["From tweets to polls: Linking text sentiment to public opinion time series".](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM10/paper/viewFile/1536/1842)
Iyyer et al. 2014. ["Political Ideology Detection Using Recursive Neural Networks".](http://www.aclweb.org/anthology/P14-1105)
Smith et al. 2010. ["Shedding (a thousand points of) light on biased language".](http://dl.acm.org/citation.cfm?id=1866719)
Charkaborty et al. 2016. ["Stop Clickbait : Detecting and Preventing Clickbaits in Online News Media".](http://ieeexplore.ieee.org/document/7752207/)
Biyani et al. 2016. ["8 Amazing Secrets for Getting More Clicks".](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11807)
Card et al. 2015. ["The Media Frames Corpus : Annotations of Frames Across Issues"](http://www.aclweb.org/anthology/P15-2072)

