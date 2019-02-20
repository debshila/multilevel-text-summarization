#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 23:02:02 2019
This script includes all the functions necessary for topic modeling.
@author: dbm
"""

# general imports
import warnings
import nltk
import pandas as pd
import numpy as np
import scipy as sp
from time import time, sleep
import json
import requests
import random
import os
import matplotlib.pyplot as plt
import re
from sklearn.externals import joblib
# Clean and lemmatize text
from nltk.corpus import stopwords
from gensim.utils import smart_open, simple_preprocess
from gensim.test.utils import datapath
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

import html5lib
import re
from bs4 import BeautifulSoup
import requests
from pprint import pprint
from pickle import dump,load
# Gensim
from gensim import similarities
import gensim
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.summarization.summarizer import summarize
from pprint import pprint

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(0)

###############################################################################
def compute_coherence_lda(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            random_state=0,
            num_topics=num_topics,
            id2word=dictionary,
#             minimum_probability=0.3, 
            alpha = 'auto', 
            eta = 'auto')
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts): 
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series(
                        [int(topic_num),
                         round(prop_topic, 4), topic_keywords]),
                    ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 
                              'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


# For new documents, extract the keywords
def get_topics_4_new_docs(model, topics,  prob = .3):
    topics.sort(key=lambda x: x[1], reverse = True) #This is in-place
    topic_keywords = [[topic[0], model.show_topic(topic[0])] for topic in topics if topic[1] > prob]
    return topic_keywords


# Display topics as a dataframe
def display_topics(feature_names, no_top_words, topic_words):
    word_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        #         print(topic_idx)
        #         print ([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        word_dict['Topic ' + str(topic_idx)] = [
            feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]
        ]
    return pd.DataFrame(
        word_dict
    )  #, columns=['Topic ' + str(k) for k in range(len(model.components_))])


# For new documents, extract the keywords
def get_similar_docs_4_new_docs(similarities_val, similarity_threshold):
    similarities_val.sort(key=lambda x: x[1], reverse = True) #This is in-place
#     print(similarities_val)
    similarities_val_thresh = [val[0] for val in similarities_val if val[1] > similarity_threshold]
    return similarities_val_thresh

def read_data_from_url(link = 'https://www.manulife.com/en/privacy-policy/privacy-statement.html', pat = '[^a-zA-z0-9.?! ]+', filt_len = 6):
    r  = requests.get(link)#'https://machinebox.io/privacy'https://www.manulife.com/en/privacy-policy/privacy-statement.html
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')    
    text_1 = soup.findAll(text=True)
    #Preprocess text
    tmp = sent_tokenize(' '.join(text_1))
    word_count_orig = len(simple_preprocess(' '.join(tmp))) 
    print(f'There are {word_count_orig} words in this document.')
    #Keep only sentences
    tmp_sent = [re.sub(pat, '', i).rstrip() for i in tmp]
    tmp_sent = [
        re.sub(pattern='[ \t]{2,}', repl=' ', string=i) for i in tmp_sent
        if len(i) > filt_len
    ]
    # Join cleaned sentences together
    txt = ' '.join(tmp_sent)
    return txt
