# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:55:20 2018

@author: Tim George Kabue
"""

import numpy as np

from keras.datasets import imdb
from keras import models,layers
from keras import preprocessing

#Number of tokens to consider for embedding layer.
tokens = 10000

#Length of integer sequences.
seq_length = 20

#Load data as lists of integers.
(train_samples,train_labels),(test_samples,test_labels) = imdb.load_data(num_words=tokens)

#Reshape lists of integers to 2Dtensor of shape (samples,seq_length)
train_samples = preprocessing.sequence.pad_sequences(train_samples,maxlen=seq_length)
test_samples = preprocessing.sequence.pad_sequences(test_samples,maxlen=seq_length)

#Start.
model = models.Sequential()

#Embedding layer.
model.add(layers.Embedding(tokens,8,input_length=seq_length))

#Network architecture.
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))

#Compilation.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Summary.
model.summary()

#Training.
history = model.fit(train_samples,train_labels,epochs=10,batch_size=32,validation_split=0.2)