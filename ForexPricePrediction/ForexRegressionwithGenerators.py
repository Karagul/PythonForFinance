# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:43:00 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +254706762054
@title: GBPNZD close price prediction
"""

import pandas as pd
import numpy as np
from keras import models,layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#******Load data.******
#------------------------------------------------------------------------------------------------

location = "C:\\Users\\User\\Projects\\MT4\\Data\\GBPNZD60.csv"
df = pd.read_csv(location,names=['Date','Time','Open','High','Low','Close','Volume'],parse_dates=[['Date','Time']],index_col='Date_Time')
data = np.asarray(df,dtype=np.float64)

#Normalize data.
scaler = MinMaxScaler()
scaler1 = MinMaxScaler()

data = scaler.fit_transform(data)
#Extract close prices which are the targets.
df_close = df['Close']
close = df_close.values.reshape(len(df_close),1)
close = scaler1.fit_transform(close)

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
            targets[j] = data[rows[j] + delay][3];
        yield samples, targets

#Create generators for train,validation and test data.
lookback = 24
step = 1
delay = 1
batch_size = 128

train_gen = generator(data,lookback=lookback,delay=delay,min_index=0,max_index=15000,shuffle=False,step=step,batch_size=batch_size)
val_gen = generator(data,lookback=lookback,delay=delay,min_index=15001,max_index=25000,shuffle=False,step=step,batch_size=batch_size)
test_gen = generator(data,lookback=lookback,delay=delay,min_index=25001,max_index=None,shuffle=False,step=step,batch_size=batch_size)

val_steps = (25000-15001-lookback) // batch_size
test_steps = (len(data)-25001-lookback) // batch_size
#------------------------------------------------------------------------------------------------

#Network architecture.
model = models.Sequential()
model.add(layers.Flatten(input_shape=(lookback//step,data.shape[-1])))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1))

model.summary()

#Network configuration.
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

#Fit model.
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mean_absolute_error']                       
val_mae = history.history['val_mean_absolute_error']

#Evaluation.
eval_values = model.evaluate_generator(test_gen, steps=test_steps)

predictions = model.predict_generator(test_gen, steps=test_steps)
predictions = scaler1.inverse_transform(predictions)
close = scaler1.inverse_transform(close)

plt.show