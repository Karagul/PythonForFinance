# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:36:55 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +254706762054
@title: Dense network candlestick predictor
"""

import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt
import talib

#Load data and split it into training and test sets.
dataset = np.load('latest_dataset.npy')
datalabels = np.load('latest_datalabels.npy')

samples = len(dataset)

train_data = dataset[19:samples-1000]; train_labels = datalabels[19:samples-1000]
test_data = dataset[samples-1000:]; test_labels = datalabels[samples-1000:]

#Feature-wise normalization/Preprocess data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

#Network architecture.
model = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Network configuration.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Network training.
no_of_epochs = 20
batch = 128
history = model.fit(train_data,train_labels,epochs=no_of_epochs,batch_size=batch,validation_split=0.2)
history_values = history.history
val_acc = history_values['val_acc']

val_acc = np.asarray(val_acc)
EMA_val_acc = talib.EMA(val_acc,10)

#Plots.
fig1 = plt.figure(1)
fig1.suptitle("Validation Accuracy vs Epochs")

axes1 = fig1.add_subplot(2,1,1)
axes1.plot(np.arange(1,len(val_acc)+1),val_acc,'b',label='Validation Accuracy')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Validation accuracy')
axes1.legend()

axes2 = fig1.add_subplot(2,1,2)
axes2.plot(np.arange(1,len(EMA_val_acc)+1),EMA_val_acc,'b',label='EMA_Validation Accuracy')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('EMA_Validation accuracy')
axes2.legend()

plt.show()