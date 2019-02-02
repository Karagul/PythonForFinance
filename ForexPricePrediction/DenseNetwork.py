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
location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60_processed.csv"
df = pd.read_csv(location)
df.replace(np.nan,0,inplace=True)
df.drop(['Date',
     'Time',
     'Open',
     'High',
     'Low',
     'Volume',
     'EMA_10',
     'EMA_20',
     'PivotPoint',
     'Crossover'], axis=1,inplace=True)

#Extract dates and time from plotting later on.
df_datetime = df.index

#Convert data to array.
data = np.asarray(df,dtype=np.float64) 

#Normalize the data.
scaler = MinMaxScaler(feature_range=(0,1))
scaler1 = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

#Extract close prices which are the targets.
df_close = df['Close']
close = df_close.values.reshape(len(df_close),1)
close = scaler1.fit_transform(close)

#Split data into train and test sets.
train,test = train_test_split(data,shuffle=False)
train_close,test_close = train_test_split(close,shuffle=False)
train_datetime,test_datetime = train_test_split(df_datetime,shuffle=False)

#Convert data to timeseries with a lookback of 5timesteps. i.e use last 5hours to predict price.
def create_timeseries(data,lookback):
    samples,targets = [],[]
    for index in np.arange(len(data)-lookback-1):
       item = data[index:(index+lookback),[0,1,2]] #item = data[index:(index+lookback),0] 
       samples.append(item)
       targets.append(data[index+lookback,0]) #targets.append(data[index+lookback,0]) 
    return np.array(samples),np.array(targets)

lookback = 20
train_samples,train_targets = create_timeseries(train,lookback)
test_samples,test_targets = create_timeseries(test,lookback)

#Network architecture.
model = models.Sequential()

model.add(layers.Bidirectional(layers.LSTM(512),input_shape=(lookback,train_samples.shape[-1])))

model.add(layers.Dense(1))

#Network configuration.
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

#Fit model.
no_of_epochs = 20
batchsize = 128
history = model.fit(train_samples,train_targets,epochs=no_of_epochs,batch_size=batchsize,validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mean_absolute_error']                       
val_mae = history.history['val_mean_absolute_error']

#Evaluation.
train_predictions = model.predict(train_samples)
test_predictions = model.predict(test_samples)

train_predictions = scaler1.inverse_transform(train_predictions)
test_predictions = scaler1.inverse_transform(test_predictions)

train_close = scaler1.inverse_transform(train_close)
test_close = scaler1.inverse_transform(test_close)

plt.plot(test_datetime,test_close,'go')
plt.plot(test_datetime[lookback+1:],test_predictions,'bo')
plt.plot(test_datetime,test_close,'g')
plt.plot(test_datetime[lookback+1:],test_predictions,'b')

#Plotting.
epochs = range(len(loss))

#Plot.
fig = plt.figure(2)
fig.suptitle('Accuracy and Loss')

axes1 = fig.add_subplot(2,1,1)
axes1.plot(np.arange(1,len(mae)+1),mae,'bo',label='Mean Absolute Error')
axes1.plot(np.arange(1,len(val_mae)+1),val_mae,'r',label='Validation_MAE')
axes1.set_xlabel('Epoch')
axes1.set_ylabel('')
axes1.legend()

axes2 = fig.add_subplot(2,1,2)
axes2.plot(np.arange(1,len(val_loss)+1),val_loss,'r',label='Validation Loss')
axes2.plot(np.arange(1,len(loss)+1),loss,'bo',label='Loss')
axes2.set_xlabel('Epoch')
axes2.set_ylabel('Loss')
axes2.legend()

#plt.show()