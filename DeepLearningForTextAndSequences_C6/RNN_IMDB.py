# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:06:25 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +254706762054
@title: LSTM with IMDB
"""

from keras import preprocessing,models,layers
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

#Get data.
num_of_words = 10000
(train_features,train_labels),(test_features,test_labels) = imdb.load_data(num_words=num_of_words)

#Preprocess the data.
max_length = 500
train_features = preprocessing.sequence.pad_sequences(train_features,maxlen=max_length)
test_features = preprocessing.sequence.pad_sequences(test_features,maxlen=max_length)

#Network architecture.
model = models.Sequential()

vector_size = 32
model.add(layers.Embedding(num_of_words,vector_size))

output_dimensionality = 32
model.add(layers.LSTM(output_dimensionality))
model.add(layers.Dense(1,activation='sigmoid'))

#Network configuration/ Compilation.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Fit model.
num_of_epochs = 100
batchSize = 128
val_size = 0.2
history = model.fit(train_features,train_labels,epochs=num_of_epochs,batch_size=batchSize,validation_split=val_size)
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
acc = history.history['acc']
loss = history.history['loss']

#Evaluate model.
eval_loss,eval_acc = model.evaluate(test_features,test_labels)

#Plot.
fig = plt.figure(1)
fig.suptitle('Accuracy and Loss')

axes1 = fig.add_subplot(2,1,1)
axes1.plot(np.arange(1,len(val_acc)+1),val_acc,'r',label='Validation Accuracy')
axes1.plot(np.arange(1,len(acc)+1),acc,'bo',label='Accuracy')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Accuracy')
axes1.legend()

axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,len(val_loss)+1),val_loss,'r',label='Validation Loss')
axes2.plot(np.arange(1,len(loss)+1),loss,'bo',label='Loss')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('Loss')
axes2.legend()

plt.show()