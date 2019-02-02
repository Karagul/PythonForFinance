# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:14:09 2019

@author: User
"""

import os
import numpy as np
import pandas as pd

directory = "C:\\Users\\User\\Projects\\Forex\\PythonForFinance\\DeepLearningForTextAndSequences_C6"
file_name = os.path.join(directory,'jena_climate_2009_2016.csv' )

file = open(file_name)
data = file.read()
file.close()

lines = data.split('\n')
header = lines[0].split(',')

print('Header: ',header)
lines = lines[1:]
print('/nNo. of lines:',len(lines))

float_data = np.zeros((len(lines),len(header)-1))

for counter,line in enumerate(lines):
    values = [float(value) for value in line.split(',')[1:]]
    float_data[counter,:] = values

#Pandas.
df = pd.read_csv(file_name)
df_float = np.asarray(df.drop(columns=list(df)[0]),dtype=np.float64)