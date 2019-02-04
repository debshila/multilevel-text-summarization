#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:50:09 2019

@author: dbm
"""

from selenium import webdriver
driver = webdriver.Chrome()
//import requests
url = "https://cnx.org/contents/AgQDEnLI@12.1:XZe6d2Jr@11/1-1-What-Is-Sociology"
driver.get(url)
el=driver.find_element_by_tag_name("body")
print(el.text)
#driver.close()