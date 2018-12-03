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
from keras import models,layers
import matplotlib.pyplot as plt


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

#******Extract features.******
train_features,train_labels = extract_features(train_dir,2000)        
val_features,val_labels = extract_features(validation_dir,1000)
test_features,test_labels = extract_features(test_dir,1000)

#******Reshape for classifier.******
train_features = np.reshape(train_features,(2000,4*4*512))
val_features = np.reshape(val_features,(1000,4*4*512))
test_features = np.reshape(test_features,(1000,4*4*512))

#******Classifier.******
model = models.Sequential()

model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])

history = model.fit(train_features,train_labels,batch_size=20,epochs=30,validation_data=(val_features,val_labels))
model.save('FeatureExtractionWithoutDataAugmentation.h5')
#model = models.load_model('FeatureExtractionWithoutDataAugmentation.h5')

#******Plot.******
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = np.arange(1,len(acc)+1)

fig = plt.figure(1)
fig.suptitle('Feature Extraction without Data Augmentation')

acc_chart = plt.subplot2grid((2,3),(0,0),colspan=3)
acc_chart.plot(epochs,acc,'bo',label='Accuracy')
acc_chart.plot(epochs,val_acc,'b',label='Validation accuracy')
acc_chart.title.set_text('Accuracy vs Number of epochs')
acc_chart.set_xlabel('Epochs')
acc_chart.set_ylabel('Accuracy')
plt.legend()

loss_chart = plt.subplot2grid((2,3),(1,0),colspan=3,sharex=acc_chart)
loss_chart.plot(epochs,loss,'ro',label='Loss')
loss_chart.plot(epochs,val_loss,'r',label='Validation loss')
loss_chart.set_title('Loss vs Number of epochs')
loss_chart.set_xlabel('Epochs')
loss_chart.set_ylabel('Loss')

plt.tight_layout()
plt.legend()

plt.show()
