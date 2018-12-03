# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:09:53 2018

@author: Tim George Kabue
@phone_number: +2540706762054
@email: timkabue@yahoo.com

"""

import numpy as np
from keras.datasets import imdb
from keras import models,layers,preprocessing
import matplotlib.pyplot as plt

#Load the data.
(train_samples,train_labels),(test_samples,test_labels) = imdb.load_data(num_words=10000)

#Preprocessing data.
train_samples = preprocessing.sequence.pad_sequences(train_samples,2494)
test_samples = preprocessing.sequence.pad_sequences(test_samples,2494)

train_labels = np.asarray(train_labels).astype(np.float32)
test_labels = np.asarray(test_labels).astype(np.float32)

#Network architecture.
model = models.Sequential()

model.add(layers.Embedding(10000,32,input_length=2494))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Network configuration/COmpilation.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Training.
history = model.fit(train_samples,train_labels,epochs=5,batch_size=512,validation_split=0.2)

history_values = history.history
val_loss = history_values['val_loss']
val_acc = history_values['val_acc']

#******Plots.******
#Validation loss and accuracy vs number of epochs.
fig1 = plt.figure(1) #Create figure.
fig1.suptitle("Validation loss and Validation accuracy vs Number of epochs") #Figure title.

axes1 = fig1.add_subplot(2,1,1)#Create subplot.
axes1.plot(np.arange(1,6),val_loss,'r',label='Validation loss')
axes1.legend(loc='upper right')
axes1.set_xlabel('Epochs')

axes2 = fig1.add_subplot(2,1,2)
axes2.plot(np.arange(1,6),val_acc,'b',label='Validation accuracy')
axes2.legend(loc='upper right')

plt.show()