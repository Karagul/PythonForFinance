# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 01:11:54 2018

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt

#Data.
y = np.random.randint(1,10,size=10)
x = np.arange(len(y))

#Create figure and add subplot.
fig1 = plt.figure(1)
fig1.suptitle('Figure 1')
my_subplot = fig1.add_subplot(1,1,1)
my_subplot.plot(x,y,color='blue')

#Event handling.
def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))

        
cid = fig1.canvas.mpl_connect('button_press_event', onclick)