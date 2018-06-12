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
test_labels = data[:200]

#******Preprocess the data******
'''Feature-wise normalization.'''
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
test_data -= mean

train_data /= std
test_data /= std

#******Network architecture.******
'''Due to K-fold validation we require multiple model instantiations.'''
def build_model():
    model = models.Sequential()
    
    model.add(layers.Dense(64,activation='relu',input_shape=(8,)))
    model.add(layers.Dense(64,activation='relu'))
    
    model.add(layers.Dense(1,activation='sigmoid'))
    
    return model

#******Compilation******
'''K-fold validation.'''
#np.random.shuffle(train_data)
no_of_folds = 4
size_of_fold = int(len(train_data)/no_of_folds)
num_epochs = 10
all_histories = []


