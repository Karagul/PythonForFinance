# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:53:45 2018

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np

#******Data.******
no_of_datapoints = 10
y1=[];y2=[];y3=[];y4=[];y5=[];y6=[];
y_all_lists = [y1,y2,y3,y4,y5,y6]

for y_list in y_all_lists:
    nos = np.random.randint(1,50,size=no_of_datapoints)
    for value in nos:
        y_list.append(value)

x = np.arange(no_of_datapoints) #x = np.arange(len(y1))

#******Create figures.******
fig1 = plt.figure(1)
fig2 = plt.figure(2)

fig1.suptitle("Figure 1")
fig2.suptitle("Figure 2")

#******Add subplots to figures.******
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)

ax3 = fig2.add_subplot(2,2,1)
ax4 = fig2.add_subplot(2,2,2)
ax5 = fig2.add_subplot(2,2,3)
ax6 = fig2.add_subplot(2,2,4)

#******Plot data points.******
ax1.plot(x,y1,"red")
ax2.plot(x,y2,"yellow")

ax3.plot(x,y3,"green")
ax4.plot(x,y4,"blue")
ax5.plot(x,y5,"indigo")
ax6.plot(x,y6,"violet")

#******Showcase.******
plt.show()