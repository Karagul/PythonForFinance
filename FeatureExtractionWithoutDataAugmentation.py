# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:41:00 2018

@author: Tim George Kabue
@email: timkabue@yahoo.com
@phone_number: +2540706762054
"""

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

#***Convolutional base.***
conv_base = VGG16(include_top=False,input_shape=(150,150,3),weights='imagenet')

#******Locations.******
base_dir = "C:\\Users\\User\\Projects\\ML_Data\\cats_dogs_small"
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

#******Generator instance.******
datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20 #Batch size...

#******Feature extraction without data augmentation function.******
def extract_features(directory,sample_count):
    #Holder for features.
    features = np.zeros(shape=(sample_count,4,4,512))
    #Holder for labels.
    labels = np.zeros(shape=(sample_count))
    #Generator itself.
    generator = datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')
    #Extract features.
    i = 0 #counter
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch #Fill in features.
        labels[i*batch_size:(i+1)*batch_size] = labels_batch #Fill in labels.
        i += 1 #Update counter.
        #Break loop if generator finishes the images.
        if i*batch_size >= sample_count:
            break
    return features,labels
        