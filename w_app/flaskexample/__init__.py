#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:39:24 2019

@author: dbm
"""
from flask import Flask
app = Flask(__name__, template_folder='template')
from flaskexample import views
