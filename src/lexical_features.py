from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from textstat.textstat import textstat
from pycorenlp import StanfordCoreNLP

import numpy as np
import pandas as pd

from collections import Counter
import os
import csv
import util
import json



def generate_lexical_features():    
    print("Generating lexical features")
    # lexical_feats = [lexical_richness, lexical_density, avg_word_length, \
                    # textstat.flesch_reading_ease, length_of_line, \
                    # compound_sentiment, positive_sentiment, negative_sentiment, \
                    # num_hyperbolic_words]
    lexical_feats = [clickbait_words_count, starts_with_number, starts_with_adverb, \
                    count_adverbs, count_adjectives, count_superlatives, count_uppercase, \
                    count_verbs, count_nouns, count_pronouns, count_prepositions, \
                    count_interjections, formality_measure, positive_sentiment_words, \
                    negative_sentiment_words] 

    #lexical_feats.extend([avg_sentence_length])

    for ng in util.newsgroups:
        print("Analyzing", ng)

        #with open(util.lexical_features_file % ng, 'w', newline='') as f:
        with open(util.home_dir + '/features/%s_more_lexical_features.csv' % ng, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['status_id'] + [x.__name__ for x in lexical_feats])

            all_link_data = pd.read_csv(util.datafile % ng, sep=',', encoding='utf8', names=util.header_names)
            
            for i, link_data in all_link_data.iterrows():
                # link_name is the *headline* of the link            
                headline = link_data['link_name']
                
                if i == 0 or pd.isnull(headline):
                    # iterrows for some reason includes the names line; link sometimes somehow
                    # doesn't have an associated name
                    continue
                if i % 1000 == 0:
                    print("...%s links complete" % i)

                global pos_headline, pos_counts
                pos_headline = nltk.pos_tag(nltk.word_tokenize(headline))
                pos_counts = compute_pos_counts()

                w.writerow([link_data['status_id']] + [str(f(headline)) for f in lexical_feats])
    
    return



def load_downworthy_dictionary():
    # taken from https://github.com/snipe/downworthy/
    downworthy_dictionary = None
    with open(util.home_dir + '/data/downworthy_dictionary.json') as f:
        downworthy_dictionary = json.load(f)   
    return set(w for w in downworthy_dictionary['replacements'])


## FEATURES
def lexical_richness(h): 
    h = h.split()
    return len(set(h)) / (1.0 * len(h))

def lexical_density(h):
    h = h.split()
    return sum(1 for word in h if word not in \
            set(stopwords.words('english'))) / (1.0 * len(h))

def avg_word_length(h):
    h = h.split()
    return sum(len(word) for word in h) / (1.0 * len(h))

def avg_sentence_length(h):
    sentences = sent_tokenize(h)
    return sum(len(s) for s in sentences) / (1.0 * len(sentences))

def length_of_line(h):
    return len(h)

#TODO why did I do SIA for this and CoreNLP for num_hyperbolic? also I should
# save the SIA scores as a global var then
def compound_sentiment(h):
    return sia.polarity_scores(h)['compound']

def positive_sentiment(h):
    return sia.polarity_scores(h)['pos']

def negative_sentiment(h):
    return sia.polarity_scores(h)['neg']

def num_hyperbolic_words(h):
    count_hyperbolic = 0
    res = nlp.annotate(h.replace(' ', '. '),    # count sentiment value of each word
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 5000,
                   })
    try:
        for s in res["sentences"]:
            sentimentValue = s["sentimentValue"] 
            if int(sentimentValue) > 3 or int(sentimentValue) < 1:
                # very negative or very positive
                count_hyperbolic += 1
    except TypeError:
        print("Error: NLP annotated output - " + res)
    
    return count_hyperbolic


def clickbait_words_count(h):
    return sum(1 for w in downworthy_set if w in h)

def starts_with_number(h):
    # pos_headline[1][1] to take care of e.g. 'The 9 Shows You Must Watch'
    return int(pos_headline[0][1] == 'CD' or pos_headline[1][1] == 'CD')

def starts_with_adverb(h):
    start = pos_headline[0][0]
    return int(start == 'RB' or start == 'RBR' or start == 'RBS')

def count_adverbs(h):
    return pos_counts['adv']

def count_adjectives(h):
    return pos_counts['adj']    

def count_pronouns(h):
    return pos_counts['pronoun'] 

def count_nouns(h):
    return pos_counts['noun'] 

def count_verbs(h):
    return pos_counts['verb'] 

def count_prepositions(h):
    return pos_counts['prep'] 

def count_interjections(h):
    return pos_counts['int'] 

def count_superlatives(h):
    return sum(1 for x in pos_headline if x[1] == 'JJS' or x[1] == 'RBS')

def count_uppercase(h):
    return sum(1 for x in h.split() if h.isupper())

def positive_sentiment_words(h):
    return sum(sia.polarity_scores(w)['pos'] for w in h.split())

def negative_sentiment_words(h):
    return sum(sia.polarity_scores(w)['neg'] for w in h.split())


# Formality measure (fmeasure) (Heylighen and Dewaele 1999): The score is used to calculate the 
# degree of formality of a text by measuring the amount of different part-of-speech tags in it. 
# It is computed as (nounfreq + adjectivefreq + prepositionfreq + articlefreq−pronounfreq−verbfreq
# −adverbfreq− interjectionfreq +100)/2. (Biyani et al. 2016)
def formality_measure(h):
    len_pos_headline = len(pos_headline)
    pos_freqs = {k: v / len_pos_headline for k, v in pos_counts.items()}
    return (pos_freqs['noun'] + pos_freqs['adj'] + pos_freqs['prep'] + pos_freqs['art'] \
            -  pos_freqs['pronoun'] - pos_freqs['verb'] - pos_freqs['adv'] - pos_freqs['int'] \
            + 100) / 2.0


def compute_pos_counts():
    # would have done a Counter but it doesn't solve the KeyError issure in the 
    # formality_measure function
    freq = {k: 0 for k in ['noun', 'adj', 'pronoun', 'verb', 'adv', 'prep', 'art', 'int']}

    def update_freq(pos, tag, tag_english):
        if pos.startswith(tag):
            if tag_english in freq:
                freq[tag_english] += 1
            return True
        return False

    possible_poses = [('NN', 'noun'), ('JJ', 'adj'), ('PR', 'pronoun'), \
            ('VB', 'verb'), ('R', 'adv'), ('IN', 'prep'), ('DT', 'art'), ('UH', 'int')]

    for w, pos in pos_headline:
        for possible_pos, possible_name in possible_poses:
            if update_freq(pos, possible_pos, possible_name):
                break

    # TODO check articles, prepositions, interjections counts
    return freq



if __name__ == "__main__":
    # load libraries for features
    sia = SIA()
    nlp = StanfordCoreNLP('http://localhost:9000')
    downworthy_set = load_downworthy_dictionary()

    # store some necessary global variables
    pos_headline = None
    pos_counts = None

    generate_lexical_features()
