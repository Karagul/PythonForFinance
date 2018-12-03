# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 21:59:20 2018

@author: User
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models,layers

import numpy as np

'''import tkinter as tk'''

import matplotlib.pyplot as plt
'''
#GUI: Root window.
root = tk.Tk()
'''
#Load data, both training set and test set.
(train_images,train_labels),(test_images,test_labels) = mnist.load_data() 

#Preprocess the data into a more suitable form.
train_images = train_images.reshape((60000,28*28)); train_images = train_images.astype(np.float32)/255;
train_labels = to_categorical(train_labels)

test_images = test_images.reshape((10000,28*28)); test_images = test_images.astype(np.float32)/255;
test_labels = to_categorical(test_labels)

#Network architecture.
model = models.Sequential()

model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))

#Compilation. Get the netowrk ready for training.
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#Training.
history = model.fit(train_images,train_labels,epochs=5,batch_size=128)
'''
#Evaluation.
def evaluation(samples,labels):
    loss,acc = model.evaluate(samples,labels)
    print("***Test accuracy: ",acc," ***")
'''
#Plot
values = history.history
y1 = values['acc']
y2 = values['loss']
x = np.arange(len(y1))

fig = plt.figure(1)
fig.suptitle("Performance: Accuracy and loss vs Number of epochs")

axes1 = fig.add_subplot(2,1,1)
axes1.plot(x,y1,'blue')

axes2 = fig.add_subplot(2,1,2)
axes2.plot(x,y2,'red')
'''
#GUI: widgets.
button = tk.Button(root,text="Evaluate model",command=lambda:evaluation(test_images,test_labels))

#GUI: gemoetry manager
button.pack()

#GUI: main loop.
root.mainloop()
'''