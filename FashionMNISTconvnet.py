# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 23:44:27 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import numpy as np

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models,layers

import matplotlib.pyplot as plt

#******Load dataset.******
(train_data,train_labels) , (test_data,test_labels) = fashion_mnist.load_data()

#******Preprocess the data.******
'''Convnets accept 3D tensors of shape - two spatial axis i.e height and width and one depth axis/channel axis. 
   For gray scale images the depth axis is 1.
'''
train_data = train_data.reshape((len(train_data),28,28,1))
test_data = test_data.reshape((len(test_data),28,28,1))

train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32')/255

'''Vectorize the labels.'''
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#******Network architecture.******
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

#******Compilation.******
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

#******Training.******
num_epochs = 10
history = model.fit(train_data,train_labels,epochs=num_epochs,batch_size=64)
metrics = history.history

#******Evaluation.******
results = model.evaluate(test_data,test_labels)