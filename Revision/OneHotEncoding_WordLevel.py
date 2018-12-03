# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:54:22 2018

@author: User
"""

import numpy as np

#Data.
samples = ['The cat sat on the mat.','The dog ate my homework.']

#Dictionary to hold indeces for the vocabulary.
index = {}

#Strip individual sample.
for sample in samples:
    #Strip individual word.
    for word in sample.split():
        if word not in index:
            index[word] = len(index) + 1 #Every word(key in dictionary) gets a index(value in dictionary: 1 to 10 or more)
        
#We'll consider only the first 10 words in each sample. 
max_length = 10

#Create tensor to hold the index.
results = np.zeros((len(samples),max_length,max(index.values())+1)) #2samples,10rows,11columns rows=words,each word is a row, columns=index

for counter_sample, sample in enumerate(samples):
    #print("counter_sample: ",counter_sample,"sample: ",sample)
    for counter_word, word in list(enumerate(sample.split()))[:max_length]:
        #print("counter_word: ",counter_word,"sample: ",word)
        word_index = index.get(word)
        results[counter_sample,counter_word,word_index] = 1.