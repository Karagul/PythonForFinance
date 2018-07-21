# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 01:59:52 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import pandas as pd
import numpy as np
from keras import models,layers

import matplotlib.pyplot as plt

#******Load data.******
df = pd.read_csv("C:/Users/User/Projects/Keras/pima-indians-diabetes.csv",
                 names=['a','b','c','d','e','f','g','h','labels'])

'''Convert dataframe to numpy array.'''
data = np.array(df[['a','b','c','d','e','f','g','h']])
labels = np.array(df['labels'])

'''Partition data to training set and test set.'''
train_data = data[200:]
train_labels = labels[200:]

test_data = data[:200]
test_labels = labels[:200]

#******Preprocess the data******
'''Feature-wise normalization.'''
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
test_data -= mean

train_data /= std
test_data /= std

#******Network architecture & Compilation.******
'''Due to K-fold validation we require multiple model instantiations.'''
def build_model():
    model = models.Sequential()
    
    model.add(layers.Dense(16,activation='relu',input_shape=(8,)))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8,activation='relu'))
    #model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc',])
    
    return model

#******Training and validation.******
'''K-fold validation.'''
#np.random.shuffle(train_data)
no_of_folds = 4
size_of_fold = int(len(train_data)/no_of_folds)
num_of_epochs = 75
all_val_acc_histories = []

for fold in range(no_of_folds):
    print("------Processing fold: ",fold,"  ------")
    
    val_data = train_data[fold*size_of_fold : (fold+1)*size_of_fold] # 0:100,100:200,200:300
    val_labels = train_labels[fold*size_of_fold : (fold+1)*size_of_fold]
    
    partial_train_data = np.concatenate([train_data[(fold+1)*size_of_fold:] , train_data[:fold*size_of_fold]])# 100:500, 200:500|0:100,300:500|0:200
    partial_train_labels = np.concatenate([train_labels[(fold+1)*size_of_fold:] , train_labels[:fold*size_of_fold]])
    
    model = build_model()
    
    history = model.fit(partial_train_data,partial_train_labels,epochs=num_of_epochs,batch_size=10,
                        validation_data=[val_data,val_labels],verbose=0)
    
    metrics = history.history
    all_val_acc_histories.append(metrics['val_acc'])
    
'''Average validation accuracy per fold.''' #Wagwan
avg_val_acc_histories = []
val_acc_values_per_epoch =[]

for counter in range(num_of_epochs):
    for value in all_val_acc_histories: #each value is mae_history
        val_acc_values_per_epoch.append(value[counter]) #a list of values from the same index in each mae_history
    avg_value_per_epoch = np.mean(val_acc_values_per_epoch) #average of those values
    avg_val_acc_histories.append(avg_value_per_epoch)
    
avg_val_acc_histories = [np.mean([history_per_fold[epoch] for history_per_fold in all_val_acc_histories]) for epoch in range(num_of_epochs)] #Tutorial way.
    
#******Plot.******
plt.plot(range(num_of_epochs),avg_val_acc_histories,'bo',label='Average validation accuracy')

plt.xlabel('Epochs')
plt.ylabel('Average validation accuracy')
plt.title('Average validation accuracy against number of epochs.')
plt.legend()
plt.show()  

#******Evaluation.******
results = model.evaluate(test_data,test_labels)

print("Evaluation results = ",results)

#******Prediction.******
predictions = model.predict(test_data)

predictions = [round(value[0]) for value in predictions]
    


