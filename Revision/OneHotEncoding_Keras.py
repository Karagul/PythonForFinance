# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:27:35 2018

@author: User
"""

from keras.preprocessing.text import Tokenizer

#Samples.
samples = ['The cat sat on the mat.','The dog ate my homework.']

#Create tokenizer, configured to 1000 most common words.
tokenizer = Tokenizer(num_words=1000)

#Build the word index.
tokenizer.fit_on_texts(samples)

#Build the one hot encoding.
sequences = tokenizer.texts_to_sequences(samples)

one_hot_encoding = tokenizer.texts_to_matrix(samples,mode='binary')