# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:24:20 2019

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import models,layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Load data.
location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60_processed2.0.csv"
df = pd.read_csv(location)
print("*******Names of original columns:\n",list(df))
df.replace(np.nan,0,inplace=True)
df.drop(['Date',
     'Time',
     'Volume',
     'PivotPoint',
     'Crossover'], axis=1,inplace=True)
print("-------Names of altered columns:\n",list(df))

#Convert data to array.
data = np.asarray(df,dtype=np.float64) 

#Normalize the data.
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

#Split data into train and test sets.
train,test = train_test_split(data,shuffle=False)

#Convert data to timeseries with a lookback of 5timesteps. i.e use last 5hours to predict price.
def create_timeseries(data,lookback):
    samples=[];targets=[]
    max_index = len(data)-lookback-1
    for row_index in range(max_index):
        rows = data[row_index:row_index+lookback,[6,7]]
        samples.append(rows)
        labels = data[row_index+lookback,[8]]
        targets.append(labels)
    return np.array(samples),np.array(targets)

lookback = 7
train_samples,train_targets = create_timeseries(train,lookback)
test_samples,test_targets = create_timeseries(test,lookback)

#Network architecture.
model = models.Sequential()

model.add(layers.Bidirectional(layers.LSTM(512),input_shape=(lookback,train_samples.shape[-1])))
                      
model.add(layers.Dense(1,activation='sigmoid'))

#Network configuration.
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#Fit model.
no_of_epochs = 20
batchsize = 128
history = model.fit(train_samples,train_targets,epochs=no_of_epochs,batch_size=batchsize,validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']                       
val_acc = history.history['val_acc']

#Evaluation.
results = model.evaluate(test_samples,test_targets)

#Plotting.
epochs = range(len(loss))

#Plot.
fig = plt.figure(2)
fig.suptitle('Accuracy and Loss')

axes1 = fig.add_subplot(2,1,1)
axes1.plot(np.arange(1,len(acc)+1),acc,'bo',label='Accuracy')
axes1.plot(np.arange(1,len(val_acc)+1),val_acc,'r',label='Validation_Accuracy')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('')
axes1.legend()

axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,len(val_loss)+1),val_loss,'r',label='Validation Loss')
axes2.plot(np.arange(1,len(loss)+1),loss,'bo',label='Loss')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('Loss')
axes2.legend()