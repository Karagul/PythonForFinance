# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 00:40:34 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import numpy as np
import matplotlib.pyplot as plt

#Data.
y1 = np.random.randint(1,10,size=10)
y2 = np.random.randint(1,10,size=10)
y3 = np.random.randint(1,10,size=10)
y4 = np.random.randint(1,10,size=10)
y5 = np.random.randint(1,10,size=10)
y6 = np.random.randint(1,10,size=10)
y7 = np.random.randint(1,10,size=10)

x = np.arange(len(y1))

#Create a figure and add two subplots.
fig1 = plt.figure(1)
fig1.suptitle('Figure 1') #Figure title.

#Add subplots to figure and create actual plots.
my_subplot1 = fig1.add_subplot(211)
my_subplot1.plot(x,y1,color='red')

my_subplot2 = fig1.add_subplot(2,1,2)
my_subplot2.plot(x,y2,color='orange')

#Create figure and subplot.
fig2 , my_subplot3 = plt.subplots(2,2)
fig2.suptitle('Figure 2')

#Plot subplots.
my_subplot3[0,0].plot(x,y3); my_subplot3[0,0].set_title('Moja'); my_subplot3[0,0].set_ylabel('Random integers')

#Fix overlapping axis
fig2.tight_layout()