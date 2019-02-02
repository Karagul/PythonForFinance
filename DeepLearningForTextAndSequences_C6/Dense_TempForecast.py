# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 18:35:15 2019

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +254706762054
@title: Densely connected network for temperature forecast
"""

import pandas as pd
import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt

#Get data.
location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\DeepLearningForTextAndSequences_C6\\jena_climate_2009_2016.csv"
df = pd.read_csv(location)
df = df.drop(columns=list(df)[0])
data = np.asarray(df,dtype=np.float64)

#Preprocess the data.
#==============================================================================
#Normalize data.
mean = data[:200000].mean(axis=0)
data -= mean
std = data[:200000].std(axis=0)
data /= std

#Define a generator.
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
        
    i = min_index + lookback
    
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
            
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        

#Create generators for train,validation and test data.
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step,batch_size=batch_size)
val_gen = generator(data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,shuffle=False,step=step,batch_size=batch_size)
test_gen = generator(data,lookback=lookback,delay=delay,min_index=300001,max_index=None,shuffle=False,step=step,batch_size=batch_size)

val_steps = (300000-200001-lookback) // batch_size
test_steps = (len(data)-300001-lookback) // batch_size

#==============================================================================

model = models.Sequential()
model.add(layers.Flatten(input_shape=(lookback//step,data.shape[-1])))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()