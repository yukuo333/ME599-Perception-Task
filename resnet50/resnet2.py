#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense,Flatten,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[2]:


pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    
)

pretrained_model.trainable = False

# random rotation to image inputs to prevent overfitting
data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

inputs = keras.Input(shape=(224,224,3))
input_shape = (60,224,224,3)
x = data_augmentation(inputs)
x = pretrained_model(x,training=False)
x = keras.layers.Dense(1024,activation='relu')(x)
x = keras.layers.Conv2D(2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
x = keras.layers.Conv2D(2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.BatchNormalization()(x)
outputs = keras.layers.Dense(3,activation='softmax')(x)
model = keras.Model(inputs,outputs)

model.summary()


# In[3]:


img_height,img_width = 224,224
batch_size = 60
validation_split = 0.2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'trainval_IMG',
    validation_split = validation_split,
    subset = "training",
    seed = 123,
    label_mode = 'categorical',
    image_size = (img_height,img_width),
    batch_size = batch_size
)


# In[4]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'trainval_IMG',
    validation_split = validation_split,
    subset = "validation",
    seed = 123,
    label_mode = 'categorical',
    image_size = (img_height,img_width),
    batch_size = batch_size
)


# In[5]:


model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[6]:


epochs = 20
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs
)


# In[7]:


fig = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','validation'])
plt.show()


# In[8]:


import cv2
import pandas as pd

class_names = train_ds.class_names

df = pd.DataFrame(columns = ['guid/image','label'])

for file in os.listdir('test_new'):
    image=cv2.imread('test_new/'+file)
    image_resized = cv2.resize(image,(img_height,img_width))
    image = np.expand_dims(image_resized,axis=0)
    pred = model.predict(image)
    output_class = class_names[np.argmax(pred)]
    df = df.append({'guid/image':file,'label':output_class},ignore_index = True)
    
df['guid/image'] = df['guid/image'].str.replace('_','/').str.replace('.jpg','')
df.to_csv('prediction_label.csv',index=False)
    


# In[ ]:




