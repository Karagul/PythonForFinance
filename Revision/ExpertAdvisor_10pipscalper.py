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
import pandas as pd
from sklearn.utils import class_weight
import keras

#Load data and split it into training and test sets.
dataset = np.load('xyz.npy')
datalabels = np.load('15min_datalabels.npy')

start = 19
end = len(dataset)
setapart = 10000

train_data = dataset[start:end-setapart]; train_labels = datalabels[start:end-setapart]
test_data = dataset[end-setapart:]; test_labels = datalabels[end-setapart:]

#Calculate class weights.
train_labels_list = train_labels.tolist()

zeros = train_labels_list.count(0)
ones = train_labels_list.count(1)

ratio = zeros/ones

#classweights = {0:1,
 #               1:ratio*2}

#Create dataframe with date and time only.
location = "C:\\Users\\User\\Projects\\MT4\\Data\\EURUSD15_5_12_2018latest.csv"
df_original = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'])
df_datetime = df_original[['Date','Time']] 
df = df_datetime.iloc[end-setapart:]

#Feature-wise normalization/Preprocess data
'''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
'''
#Network architecture.
model = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Network configuration.
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Network training.
no_of_epochs = 100
batch = 128
classweights = class_weight.compute_class_weight('balanced',np.unique(train_labels),train_labels)
history = model.fit(train_data,train_labels,epochs=no_of_epochs,batch_size=batch,validation_split=0.2,class_weight=classweights)
history_values = history.history
val_acc = history_values['val_acc']
val_loss = history_values['val_loss']
loss = history_values['loss']
acc = history_values['acc']

val_acc = np.asarray(val_acc)
EMA_val_acc = talib.EMA(val_acc,10)

#Plots.
fig1 = plt.figure(1)
fig1.suptitle("Validation Accuracy vs Epochs")

axes1 = fig1.add_subplot(3,2,1)
axes1.plot(np.arange(1,len(val_acc)+1),val_acc,'b',label='Validation Accuracy')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Validation accuracy')
axes1.legend()

axes2 = fig1.add_subplot(3,2,2)
axes2.plot(np.arange(1,len(EMA_val_acc)+1),EMA_val_acc,'b',label='EMA_Validation Accuracy')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('EMA_Validation accuracy')
axes2.legend()

axes3 = fig1.add_subplot(3,2,3)
axes3.plot(np.arange(1,len(val_loss)+1),val_loss,'b',label='Validation loss')
axes3.set_xlabel('Epoch')
axes3.set_ylabel('Validation loss')
axes3.legend()

axes4 = fig1.add_subplot(3,2,4)
axes4.plot(np.arange(1,len(loss)+1),loss,'b',label='loss')
axes4.set_xlabel('Epoch')
axes4.set_ylabel('loss')
axes4.legend()

axes5 = fig1.add_subplot(3,2,5)
axes5.plot(np.arange(1,len(acc)+1),acc,'b',label='acc')
axes5.set_xlabel('Epoch')
axes5.set_ylabel('acc')
axes5.legend()

plt.show()

#Predictions.
predictions = model.predict_classes(test_data)

all_predictions_datetime = list(zip(df['Date'],df['Time'],predictions,test_labels))
df_all_predictions_datetime = pd.DataFrame(data=all_predictions_datetime,columns=['Date','Time','Prediction','Target'])

index = 0
correct = 0
incorrect = 0
while index <= df_all_predictions_datetime.iloc[-1].name:
    if df_all_predictions_datetime.iloc[index]['Prediction'] == df_all_predictions_datetime.iloc[index]['Target']:
        correct += 1
        print("+++Correct: ",correct)
    elif df_all_predictions_datetime.iloc[index]['Prediction'] != df_all_predictions_datetime.iloc[index]['Target']:
        incorrect += 1
        print("---Incorrect: ",incorrect)
    index += 1

wrong = np.nonzero(model.predict_classes(test_data).reshape((-1,)) != test_labels)
right = np.nonzero(model.predict_classes(test_data).reshape((-1,)) == test_labels)

wrong_rows = df_all_predictions_datetime.iloc[wrong[0]]
right_rows = df_all_predictions_datetime.iloc[right[0]] 