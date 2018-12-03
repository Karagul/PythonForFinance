# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:51:46 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers,models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

#******Directories.******
base_dir = "C:\\Users\\User\\Projects\\ML_Data\\cats_dogs_small"
train_dir = os.path.join(base_dir,"train")
val_dir = os.path.join(base_dir,"validation")
test_dir = os.path.join(base_dir,"test")

#******Data preprocessing.******
conv_base = VGG16(weights="imagenet",include_top=False,input_shape=(150,150,3))
conv_base.trainable = False

train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')


#******Network architecture.******
model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,validation_data=val_generator,validation_steps=50)

#******Plot.******
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(1,len(acc)+1)

fig = plt.figure(1)
fig.suptitle("Feature Extraction with Data Augmentation 100epochs")

acc_chart = plt.subplot2grid((2,1),(0,0))
acc_chart.plot(epochs,acc,'bo',label='Accuracy')
acc_chart.plot(epochs,val_acc,'b',label='Validation accuracy')
acc_chart.set_title('Accuracy vs Number of Epochs')
acc_chart.set_xlabel('Epochs')
acc_chart.set_ylabel('Accuracy')
plt.legend()

loss_chart = plt.subplot2grid((2,1),(1,0))
loss_chart.plot(epochs,loss,'ro',label='Loss')
loss_chart.plot(epochs,val_loss,'r',label='Validation loss')
loss_chart.set_title('Loss vs Number of Epochs')
loss_chart.set_xlabel('Epochs')
loss_chart.set_ylabel('Loss')
plt.legend()

plt.show()
