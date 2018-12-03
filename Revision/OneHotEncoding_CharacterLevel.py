# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:35:28 2018

@author: User
"""

import numpy as np
import string

#Data.
samples = ['The cat sat on the mat.','The dog ate my homework.']

#Build index of characters.
characters = string .printable

token_index = dict(zip(np.arange(1,len(characters)+1),characters))

#Results of one hot encodeing.
max_length = 62

results = np.zeros(shape=(len(samples),max_length,max(token_index.keys())+1))

for counter_sample,sample in enumerate(samples):
    for counter_char,character in enumerate(sample):
        index = token_index.get(character)
        results[counter_sample,counter_char,index] = 1.