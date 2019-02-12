"""
Created on Mon Jan 28 16:24:35 2019

@author: dbm
"""

from flaskexample import app
from flask import Flask, request, render_template, jsonify
from gensim.summarization import summarize, keywords
import gensim as gs
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer 
from nltk import sent_tokenize
# Init the Wordnet Lemmatizer
wnl = WordNetLemmatizer()


#from my_module import model

@app.route('/')
@app.route('/index')
def index():
  product = {'name': 'Gist do it!'}  
  return render_template("index.html", title = 'Home', user = product)

#@app.route('/', methods=['POST'])
#def my_form_post():
#    text = request.form['text']
#    processed_text = text.lower()
#    return processed_text
#'/predict', 
@app.route('/', methods=["POST"])
def get_text():
    # text = request.form.get('textbox')
    text = request.get_json(force=True)['text']
    #text =  mystring.replace('\n', ' ')
#    text = sent_tokenize(text)
    result = summarize(text, ratio = 0.20, split = True)
    title = summarize(text, ratio = 0.08, split = True) #

    #result = model.predict(text)
    return jsonify({'Title':title, 'Gist': result})

@app.route('/', methods=["POST"])
def script_output():
    output = summarize_text_extractive('./script')
    return output

    

#def predict(txt):
#    input_text =  ''
#    result = model.predict('model prediction')
#    return result