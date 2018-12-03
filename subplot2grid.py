# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 00:49:09 2018

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np

acc = np.random.randint(1,30,30)
val_acc = np.random.randint(1,30,30)
loss = np.random.randint(1,30,30)
val_loss = np.random.randint(1,30,30)

epochs = np.arange(1,len(acc)+1)

plt.figure()

acc_chart = plt.subplot2grid((2,3),(0,0),colspan=3)
acc_chart.plot(epochs,acc,'bo',label='Accuracy')
acc_chart.plot(epochs,val_acc,'b',label='Validation accuracy')

loss_chart = plt.subplot2grid((2,3),(1,0),colspan=3)
loss_chart.plot(epochs,loss,'ro',label='Loss')
loss_chart.plot(epochs,val_loss,'r',label='Validation loss')