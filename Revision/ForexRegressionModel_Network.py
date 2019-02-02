# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:42:45 2018

@author: User
"""

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

#Load data and split it into training and test sets.
dataset = np.load('15min_dataset.npy')
datalabels = np.load('15min_datatargets_regression.npy')

start = 19
end = len(dataset)
setapart = 1000

train_data = dataset[start:end-setapart]; train_labels = datalabels[start:end-setapart]
test_data = dataset[end-setapart:]; test_labels = datalabels[end-setapart:]


#Feature-wise normalization/Preprocess data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

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

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

#Network architecture.
model = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1))

#Network configuration.
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

#Network training.
no_of_epochs = 100
batch = 128
history = model.fit(train_data,train_labels,epochs=no_of_epochs,batch_size=batch,validation_split=0.2)
history_values = history.history
'''
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
'''
#Predictions.
predictions = model.predict(test_data)

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
