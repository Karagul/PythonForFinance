# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:00:34 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models,layers
from sklearn.model_selection import KFold
import talib

#Load data.
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

#Preprocess the data.
#*************************************************************************************

#Normalize train and target samples.
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

#*************************************************************************************

#Network architecture and configuration(comiplation.)
def build_model():
    model = models.Sequential()
    
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1)) #No activation for a regression model.
    
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    
    return model
    
#Train network - with KFold cross validation and evlauate.
num_epochs = 80
all_val_mae_histories = [] #Validation Accuracy values
num_folds = 4
kfold = KFold(num_folds) 
size_of_batch = 1

fold = 1   

for train,test in kfold.split(train_data):
    print("******Fold : ",fold,"******")
    model = build_model()
    history = model.fit(train_data[train[:]],train_targets[train[:]],epochs=num_epochs,batch_size=size_of_batch,validation_data=(train_data[test[:]],train_targets[test[:]]),verbose=1)
    history_values = history.history
    val_mae_history = history_values['val_mean_absolute_error']
    all_val_mae_histories.append(val_mae_history)
    fold += 1
'''
1. There are as many elements in each history as there are epochs. In this case 500 elements in each history such as validation accuracy.
2. There is also as many of each history as there are folds.
3. Therefore validation accuracy will have 10 elements each with 500 elements.
4. We need the average of each of the 500elements. i.e the average of each epoch in each fold.
'''
average_mae_history = [np.mean([maeHistory[epoch] for maeHistory in all_val_mae_histories]) for epoch in range(num_epochs)]

#Plots.
#***********************************************************************************

#Create figure.
fig = plt.figure(1)
fig.suptitle('Regression')

#Subplots.
axes1 = fig.add_subplot(2,1,1)
axes1.plot(np.arange(1,num_epochs+1),average_mae_history,'b',label='Mean Accuracy Error')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Validation MAE')
axes1.legend()

average_mae_history = np.asarray(average_mae_history)
ema_values = talib.EMA(average_mae_history,10)

axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,len(ema_values)+1),ema_values,'r',label='Smoothed Mean Accuracy Error')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('10period_EMA MAE')
axes2.legend()

plt.tight_layout()
plt.show()      
#***********************************************************************************