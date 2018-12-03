# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:09:49 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

from keras.datasets import reuters
from keras import preprocessing,models,layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Load data.
(train_samples,train_labels),(test_samples,test_labels) = reuters.load_data(num_words=10000)

#Preprocess the data.
max_length = max([len(sample) for sample in train_samples])

train_samples = preprocessing.sequence.pad_sequences(train_samples,max_length) #Pad all samples to have the same length.
test_samples = preprocessing.sequence.pad_sequences(test_samples,max_length)

train_labels = to_categorical(train_labels)#One hot encode the labels.
test_labels = to_categorical(test_labels)

#Network architecture
model = models.Sequential()

model.add(layers.Embedding(10000,16,input_length=max_length))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

#Configure the network for training./Compilation.
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy']) 

#Train the network.
epochs = 20
history = model.fit(train_samples,train_labels,epochs=epochs,batch_size=64,validation_split=0.2)
history_values = history.history
val_acc = history_values['val_acc']
val_loss = history_values['val_loss']
acc = history_values['acc']
loss = history_values['loss']

#Plot.
#****************************************************************************

#Create figure.
fig = plt.figure(1)
#Figure title.
fig.suptitle("Figure 1")

#Add subplots to figure.
axes1 = fig.add_subplot(2,1,1)
#Plot.
axes1.plot(np.arange(1,epochs+1),val_acc,'go',label='Validation accuracy')
#Details.
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Validation accuracy')
axes1.set_title('Validation accuracy vs Number of Epochs')
axes1.legend()

#Add subplots.
axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,epochs+1),val_loss,'g',label='Validation loss')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('Validation loss')
axes2.set_title('Validation loss vs Number of Epochs')
axes2.legend()

#******************************************************************************

#Evaluation.
results = model.evaluate(test_samples,test_labels)