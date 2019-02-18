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
import pandas as pd
import requests
import random
import os
import re
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
import xml.etree.ElementTree as etree

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

# parse each xml file and generate a df
def xml_to_df(file):
#     print(file)
    e = etree.parse(file)
    text = [e.findall('.//SUBTEXT')[i].text for i in range(len(e.findall('.//SECTION')))]
    sec_title = [e.findall('.//SUBTITLE')[i].text for i in range(len(e.findall('.//SECTION')))]
#    policy_doc = {'file':file, 'text': text, 'title': sec_title}    
    df = pd.DataFrame({'file':file, 'section':sec_title, 'text': text})
    return df




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
pp_xml = '/Users/dbm/Downloads/corpus/'

print(
    f'No. of Privacy policy Documents: {len(pp)} \nNo. of Opt out policy Documents: {len(oo)}\nNo. of privacy policy XML Documents: {len(pp_xml)}'
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

########################################
###### Process ACL XML files ###########
########################################    
# Process xml privacy files
pp_xml = '/Users/dbm/Downloads/corpus/'
pp_xml_files = ls_fullpath(directory=pp_xml)
pp_xml_files = [i for i in pp_xml_files if '.DS_Store' not in i]

pp_xml_docs = [ ]
for file in pp_xml_files:
    parsed_file = xml_to_df(file, doc_type = 'pp')
    pp_xml_docs.append(parsed_file)

pp_xml_df = pd.concat(pp_xml_docs, axis = 0)    
pp_xml_df.head(2)

#  Filter out rows with missing text
print(pp_xml_df.info())
print(pp_xml_df.isna().sum())
# pp_xml_docs = pd.concat(pp_xml_df)
pp_xml_df = pp_xml_df.reset_index()
pp_xml_df.head(2)

# Filter out rows with missing text
pp_xml_df = pp_xml_df[pp_xml_df['text'].notnull()]
dump(pp_xml_df, open("/Users/dbm/Documents/Insight S19/data/acl_privacy_policy.pkl", "wb"))

print(pp_xml_df.shape)
# Filter out the Policy names
pp_df_text = pp_xml_df[pp_xml_df['section'].notnull()]
print(pp_df_text.shape)

# Retain policy id
pp_df_name = pp_xml_df[pp_xml_df['section'].isnull()]
print(pp_df_name.shape)

## Save datasets
dump(pp_df_text, open("/Users/dbm/Documents/Insight S19/data/privacy_policy_acl_text.pkl", "wb"))
dump(pp_df_name, open("/Users/dbm/Documents/Insight S19/data/privacy_policy_acl_title.pkl", "wb"))

# Add the company names to frequent words
co_names_acl = pp_df_text['file'].apply(lambda x: extract_co_name(x)).unique()
co_names_acl = co_names_acl.tolist()
frequent_words = frequent_words + co_names_acl
frequent_words = list(set(frequent_words))
print(len(frequent_words))

# Clean, tokenize text
pp_data_words = tokenize(pp_df_text['text'], stop_words=stop_words,
                      frequent_words=frequent_words)


pp_data_words = [' '.join(word) for word in pp_data_words]
pp_data_words
print(f'type: {type(pp_data_words)}, len: {len(pp_data_words)}')

dump(pp_data_words, open("/Users/dbm/Documents/Insight S19/data/acl_privacy_policy_words.pkl", "wb"))

