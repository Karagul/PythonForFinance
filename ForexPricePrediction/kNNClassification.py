# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:27:18 2019

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors

#Load data.
location = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\ForexPricePrediction\\GBPNZD60_processed.csv"
df = pd.read_csv(location)
print("*******Names of original columns:\n",list(df))
df.replace(np.nan,0,inplace=True)
df.drop(['Date',
     'Time',
     'Volume',
     'PivotPoint',
     'Crossover'], axis=1,inplace=True)
print("-------Names of altered columns:\n",list(df))

samples = np.asarray(df.drop('Label',axis=1))
labels = np.asarray(df['Label'])

train_samples,test_samples,train_labels,test_labels = train_test_split(samples,labels,shuffle=False)

#Define classifier.
clf = neighbors.KNeighborsClassifier(n_neighbors=20)

#Fit.
clf.fit(train_samples,train_labels)

#Evaluate.
acc = clf.score(test_samples,test_labels)
print(acc)