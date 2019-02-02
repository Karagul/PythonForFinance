# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:27:18 2019

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

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

#Scale data.
scaler = MinMaxScaler(feature_range=(0,1))

data = np.asarray(df,np.float64)
data = scaler.fit_transform(data)

#Create timeseries
def create_timeseries(data,lookback):
    samples=[];targets=[]
    max_index = len(data)-lookback-1
    for row_index in range(max_index):
        rows = data[row_index:row_index+lookback,[6,7]]
        samples.append(rows)
        labels = data[row_index+lookback,[8]]
        targets.append(labels)
    return np.array(samples),np.ravel(np.array(targets))

train,test = train_test_split(data,shuffle=False)

train_samples,train_targets = create_timeseries(train,5)
test_samples,test_targets = create_timeseries(test,5)
train_samples = train_samples.reshape((train_samples.shape[0],train_samples.shape[1]*train_samples.shape[2]))
test_samples = test_samples.reshape((test_samples.shape[0],test_samples.shape[1]*test_samples.shape[2]))

#Define classifier.
clf = neighbors.KNeighborsClassifier(n_neighbors=20)

#Fit.
clf.fit(train_samples,train_targets)

#Evaluate.
acc = clf.score(test_samples,test_targets)
print(acc)