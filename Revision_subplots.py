# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:44:49 2018

@author: Tim George Kabue
@phone_number: +2540706762054
@email: timkabue@yahoo.com
"""

import matplotlib.pyplot as plt
import numpy as np

#******Create data.******
y1=[];y2=[];y3=[];y4=[];y5=[];y6=[];y7=[];y8=[];y9=[];
y=[y1,y2,y3,y4,y5,y6,y7,y8,y9]

for axis in y:
    values = np.random.randint(1,50,size=50)
    for value in values:
        axis.append(value)

#******       