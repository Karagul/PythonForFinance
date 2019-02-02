# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:51:15 2018

@author: Tim George Kabue
@phone_number: +254706762054
@email: timkabue@yahoo.com
"""

from keras.datasets import imdb
from keras import preprocessing,models,layers
import numpy as np
import matplotlib.pyplot as plt

#Load dataset.
no_of_tokens = 10000
(train_samples,train_labels),(test_samples,test_labels) = imdb.load_data(num_words=no_of_tokens)

#Preprocess the data.
max_len = 500
train_samples = preprocessing.sequence.pad_sequences(train_samples,maxlen=max_len)
test_samples = preprocessing.sequence.pad_sequences(test_samples,maxlen=max_len)

#Network architecture.
model = models.Sequential()

vector_size = 32
model.add(layers.Embedding(no_of_tokens,vector_size))

model.add(layers.SimpleRNN(32))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

#Network configuration.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#Fit model.
size_of_batch = 128
no_of_epochs = 5
history = model.fit(train_samples,train_labels,batch_size=size_of_batch,epochs=no_of_epochs)
loss = history.history['loss']
#val_loss = history.history['val_loss']
acc = history.history['acc']                       
#val_acc = history.history['val_acc']

#Evaluate model.
eval_loss,eval_acc = model.evaluate(test_samples,test_labels)

#Plot.
fig = plt.figure(1)
axes1 = fig.add_subplot(2,1,1)
axes1.plot(np.arange(1,no_of_epochs+1),loss,'r',label='loss')
#axes1.plot(np.arange(1,no_of_epochs+1),val_loss,'b',label='Validation loss')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Loss')
axes1.legend()

axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,no_of_epochs+1),acc,'r',label='accuracy')
#axes2.plot(np.arange(1,no_of_epochs+1),val_acc,'b',label='Validation accuracy')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('Accuracy')
axes2.legend()