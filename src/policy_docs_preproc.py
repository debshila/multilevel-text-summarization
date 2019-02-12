#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:30:25 2019

Description: This script reads in the privacy policy related HTML files from downloaded
from https://usableprivacy.org/data. The script reads in the each HTML file, cleans, tokenizes 
and lemmatizes the text, and saves the processed file as a pkl.

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
import re
from xml.etree import cElementTree as ET
from sklearn.externals import joblib
from pickle import dump,load

# Clean and lemmatize text
from nltk.corpus import stopwords
from gensim.utils import smart_open, simple_preprocess
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import html5lib
import re
from bs4 import BeautifulSoup
import requests

# Enable logging for gensim - optional
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(0)

# Directory paths
pp_dir = '/Users/dbm/Downloads/OPP-115/sanitized_policies/'
oo_dir = '/Users/dbm/Downloads/OptOutChoice-2017_v1.0/SanitizedPrivacyPolicies/'

########################################
############# Utilities ################
########################################    


# Get a lisst of all files within a directory
def ls_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

# Extract company name from file name
def extract_co_name(file, pat='\d+_|\.com|\.edu|\.html'):
    co_name = re.sub(pattern=pat, repl='', string=os.path.basename(file))
    return co_name

# Read in files and do preliminary preprocessing
def read_html(file, pat = '[^a-zA-z0-9.?!/ ]+', filt_len=6, doc_type='pp'):
    html_file = open(file, 'r', errors='ignore')
    source_code = html_file.read()
    soup = BeautifulSoup(source_code, 'html.parser')
    tmp = sent_tokenize(''.join(soup.findAll(text=True)))
    tmp_sent = [re.sub(pat, '', i).rstrip() for i in tmp]
    tmp_sent = [
        re.sub(pattern='[ \t]{2,}', repl=' ', string=i) for i in tmp_sent
        if len(i) > filt_len
    ]
    txt = ' '.join(tmp_sent)
#     named_entities.append(get_named_entities(txt))
    try:
        year = re.search(string=txt, pattern='20\d{2}').group()
    except AttributeError:
        year = ''  # apply your error handling
        type(year)
    data = {'file': file, 'year': year, 'doc_type': doc_type, 'text': txt}
    return data

# tokenize, lower case, and lemmatize words
def tokenize(series, stop_words, frequent_words):
    return (
        series
        .apply(lambda x: simple_preprocess(x))
        .apply(lambda tokens: [token for token in tokens if token not in stop_words])
        .apply(lambda tokens: [token for token in tokens if token not in frequent_words])
#       .apply(lambda tokens: [token for token in tokens if token not in get_named_entities(tokens)])
        .apply(lambda tokens: [wnl.lemmatize(token) for token in tokens])
        .apply(lambda tokens: [token for token in tokens if len(token) > 3])
    )

########################################
########### Read in files ##############
########################################    
pp_files = ls_fullpath(directory=pp_dir)
oo_files = ls_fullpath(directory=oo_dir)
oo_files = [i for i in oo_files if '.DS_Store' not in i]

# Extract company names from filenames
co_names_pp = [extract_co_name(file=file) for file in pp_files]
co_names_oo = [extract_co_name(file=file) for file in oo_files]
co_names = co_names_pp + co_names_oo

# Read in privacy policy and opt-out policy documents and preprocess them
pp = [read_html(file=file, doc_type='pp') for file in pp_files]
oo = [read_html(file=file, doc_type='oo') for file in oo_files]

print(
    f'No. of Privacy policy Documents: {len(pp)} \nNo. of Opt out policy Documents: {len(oo)}'
)

# Combine data into dataframe
df = pd.DataFrame(pp+oo)

# Save dataset
dump(df, open("/Users/dbm/Documents/Insight S19/data/privacy_optout_policy.pkl", "wb"))

# Words to filter out before tokenization
stop_words = set(stopwords.words('english'))
frequent_words = [
    'privacy', 'policy', 'andor', 'terms', 'service', 
    'please', 'valve','jibjab', 'steam', 'microsoft'
]
frequent_words.append(co_names)
# Initialize lemmatizer
wnl = WordNetLemmatizer()

# Clean, lemmatize, and tokenize text
data_words = tokenize(df['text'], stop_words=stop_words,
                      frequent_words=frequent_words)
data_words = [' '.join(word) for word in data_words]
print(f'type: {type(data_words)}, len: {len(data_words)}')

## Save dataset
dump(data_words, open("/Users/dbm/Documents/Insight S19/data/privacy_optout_policy_cleaned.pkl", "wb"))