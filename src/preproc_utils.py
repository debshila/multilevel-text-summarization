#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 23:29:21 2019
This script includes all the functions necessary for data preprocessing.
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
import xml.etree.ElementTree as etree
from sklearn.externals import joblib

# Clean and lemmatize text
from nltk.corpus import stopwords
from gensim.utils import smart_open, simple_preprocess
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
import html5lib
import re
from bs4 import BeautifulSoup
import requests
from pprint import pprint
from pickle import dump,load
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
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

nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore", category=DeprecationWarning)
random.seed(0)
###############################################################################

# Read in all story files
def ls_fullpath(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def extract_co_name(file, pat='\d+_|\.com|\.edu|\.html|\.xml|www_|_co'):
    co_name = re.sub(pattern=pat, repl='', string=os.path.basename(file))
    return co_name

# parse each xml file and generate a df
def xml_to_df(file, doc_type = 'pp'):
#     print(file)
    e = etree.parse(file)    
    text = [e.findall('.//SUBTEXT')[i].text for i in range(len(e.findall('.//SECTION')))]
    sec_title = [e.findall('.//SUBTITLE')[i].text for i in range(len(e.findall('.//SECTION')))]
#    policy_doc = {'file':file, 'text': text, 'title': sec_title}    
    df = pd.DataFrame({'file':file, 'section':sec_title, 'text': text, 'type':doc_type})
    return df

def xml_to_df_nosec(file, doc_type = 'pp'):
#     print(file)
    """
    Parses XML and concatenates text across sections of each document.
    """
    e = etree.parse(file)
    text = [e.findall('.//SUBTEXT')[i].text for i in range(len(e.findall('.//SECTION')))]
    text = [i for i in text if i is not None]

# Filter out the first line which usually includes the dates and other info
    tmp = sent_tokenize(text[0])
    tmp = ' '.join(tmp[1:])    
    tmp1 = [tmp] + text[1:]
    tmp1 = ' '.join(tmp1)

#     Filter out the first sentence
    df = {'file':file,'text': tmp1, 'type':doc_type}
    return df

# Read in html files and do preliminary preprocessing
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

def read_data_from_url(link, pat = '[^a-zA-z0-9.?! ]+', filt_len = 6):
    r  = requests.get(link)#'https://machinebox.io/privacy'https://www.manulife.com/en/privacy-policy/privacy-statement.html
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')    
    text_1 = soup.findAll(text=True)
    #Preprocess text
    tmp = sent_tokenize(''.join(text_1))
    word_count_orig = len(simple_preprocess(' '.join(tmp)))    
    #Keep only sentences
    tmp_sent = [re.sub(pat, '', i).rstrip() for i in tmp]
    tmp_sent = [
        re.sub(pattern='[ \t]{2,}', repl=' ', string=i) for i in tmp_sent
        if len(i) > filt_len
    ]
    # Join cleaned sentences together
    txt = ' '.join(tmp_sent)
    return txt



# load spacy pretrained model for Named Entity Recognition
def remove_names(text, ner_2_filter = ['PERSON', 'GPE', 'ORG', 'DATE','PRODUCT']):
    doc = nlp(text)
    removed_name = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ in ner_2_filter:
            removed_name.append('')
        else:
            removed_name.append(token.string)
    return ''.join(removed_name)

def extract_org_names(text):
    doc = nlp(text)
    org_name = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == "ORG":
            org_name.append(token.string)
        else:
            org_name.append('')
    org_name = set(org_name)
    return ''.join(org_name)

# tokenize, lower case, and lemmatize words
def tokenize(series, stop_words, frequent_words):
    return (
        series
        .apply(lambda x: remove_names(x))
        .apply(lambda x: simple_preprocess(x))
        .apply(lambda tokens: [token for token in tokens if token not in stop_words])
        .apply(lambda tokens: [token for token in tokens if token not in frequent_words])
#         .apply(lambda tokens: [token for token in tokens if token not in get_named_entities(tokens)])
        .apply(lambda tokens: [wnl.lemmatize(token) for token in tokens])
        .apply(lambda tokens: [token for token in tokens if token not in frequent_words])
        .apply(lambda tokens: [token for token in tokens if len(token) > 3])
    )

# Initialize lemmatizer
wnl = WordNetLemmatizer()
###############################################################################
    
    # Words to filter (stopwords from here: https://gist.github.com/sebleier/554280)
stop_words = [
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't",
    "have", "haven", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma",
    "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's",
    "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
    "than", "that", "that'll", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we",
    "were", "weren", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't",
    "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's",
    "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll",
    "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd",
    "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's",
    "would", "able", "abst", "accordance", "according", "accordingly",
    "across", "act", "actually", "added", "adj", "affected", "affecting",
    "affects", "afterwards", "ah", "almost", "alone", "along", "already",
    "also", "although", "always", "among", "amongst", "announce", "another",
    "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways",
    "anywhere", "apparently", "approximately", "arent", "arise", "around",
    "aside", "ask", "asking", "auth", "available", "away", "awfully", "b",
    "back", "became", "become", "becomes", "becoming", "beforehand", "begin",
    "beginning", "beginnings", "begins", "behind", "believe", "beside",
    "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came",
    "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com",
    "come", "comes", "contain", "containing", "contains", "couldnt", "date",
    "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg",
    "eight", "eighty", "either", "else", "elsewhere", "end", "ending",
    "enough", "especially", "et", "etc", "even", "ever", "every", "everybody",
    "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff",
    "fifth", "first", "five", "fix", "followed", "following", "follows",
    "former", "formerly", "forth", "found", "four", "furthermore", "g", "gave",
    "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes",
    "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence",
    "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid",
    "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im",
    "immediate", "immediately", "importance", "important", "inc", "indeed",
    "index", "information", "instead", "invention", "inward", "itd", "it'll",
    "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", "knows",
    "l", "largely", "last", "lately", "later", "latter", "latterly", "least",
    "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little",
    "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make",
    "makes", "many", "may", "maybe", "mean", "means", "meantime", "meanwhile",
    "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly",
    "mr", "mrs", "much", "mug", "must", "n", "na", "name", "namely", "nay",
    "nd", "near", "nearly", "necessarily", "necessary", "need", "needs",
    "neither", "never", "nevertheless", "new", "next", "nine", "ninety",
    "nobody", "non", "none", "nonetheless", "noone", "normally", "nos",
    "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often",
    "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord",
    "others", "otherwise", "outside", "overall", "owing", "p", "page", "pages",
    "part", "particular", "particularly", "past", "per", "perhaps", "placed",
    "please", "plus", "poorly", "possible", "possibly", "potentially", "pp",
    "predominantly", "present", "previously", "primarily", "probably",
    "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite",
    "qv", "r", "ran", "rather", "rd", "readily", "really", "recent",
    "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
    "relatively", "research", "respectively", "resulted", "resulting",
    "results", "right", "run", "said", "saw", "say", "saying", "says", "sec",
    "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
    "self", "selves", "sent", "seven", "several", "shall", "shed", "shes",
    "show", "showed", "shown", "showns", "shows", "significant",
    "significantly", "similar", "similarly", "since", "six", "slightly",
    "somebody", "somehow", "someone", "somethan", "something", "sometime",
    "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically",
    "specified", "specify", "specifying", "still", "stop", "strongly", "sub",
    "substantially", "successfully", "sufficiently", "suggest", "sup", "sure",
    "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks",
    "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered",
    "therefore", "therein", "there'll", "thereof", "therere", "theres",
    "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou",
    "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus",
    "til", "tip", "together", "took", "toward", "towards", "tried", "tries",
    "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately",
    "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used",
    "useful", "usefully", "usefulness", "uses", "using", "usually", "v",
    "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want",
    "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever",
    "what'll", "whats", "whence", "whenever", "whereafter", "whereas",
    "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim",
    "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos",
    "whose", "widely", "willing", "wish", "within", "without", "wont", "words",
    "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero",
    "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate",
    "appropriate", "associated", "best", "better", "c'mon", "c's", "cant",
    "changes", "clearly", "concerning", "consequently", "consider",
    "considering", "corresponding", "course", "currently", "definitely",
    "described", "despite", "entirely", "exactly", "example", "going",
    "greetings", "hello", "help", "hopefully", "ignored", "inasmuch",
    "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep",
    "keeps", "novel", "presumably", "reasonably", "second", "secondly",
    "sensible", "serious", "seriously", "sure", "t's", "third", "thorough",
    "thoroughly", "three", "well", "wonder"
]

# Words to filter
# stop_words = set(stopwords.words('english'))
frequent_words = [
    'privacy', 'profile', 'policy', 'andor', 'terms', 'service', 'product', 'provide','last','identifiable',
    'please', 'valve','jibjab', 'steam', 'microsoft', 'information', 'site', 'party','device','mail','provider',
    'member', 'media', 'person','france','wolfram','kayak','mediabistro', 'washington','globe','modified','scribd',
    'astro','indigo','ucla'
]


