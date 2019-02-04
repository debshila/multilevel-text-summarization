#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:40:25 2019

@author: dbm
"""

from flaskexample import app
from flask import render_template

@app.route('/')
@app.route('/index')
def index():
  return render_template("index.html", title = 'Home', user = {'nickname':'Gist do it'})


@app.route('/predict')
def predict():
    return "Model does not work yet."