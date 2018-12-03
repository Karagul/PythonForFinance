# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:54:26 2018

@author: User
"""

from keras.preprocessing .text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(samples)

