# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:15:40 2021

@author: harsh
"""

from flask import Flask, render_template, request

import searchEngine
import pandas as pd

app = Flask(__name__)
@app.route('/')
def hello_world():
  return 'welcome to fairKR engine'

@app.route('/suggest/<input_str>')
def suggest(input_str):
    ans = searchEngine.create_docs(input_str)
    return ans


    

if __name__ == '__main__':
  app.run()