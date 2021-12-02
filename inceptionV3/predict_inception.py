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
from tensorflow.python.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ### Set up Training and Validation Dataset

# In[2]:


img_height,img_width = 299,299
batch_size = 64
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


# In[3]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'trainval_IMG',
    validation_split = validation_split,
    subset = "validation",
    seed = 123,
    label_mode = 'categorical',
    image_size = (img_height,img_width),
    batch_size = batch_size
)


# ### import Inception model

# In[4]:


# pretrained model
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299,299,3),
    pooling='avg',
    classes=3
)

inception_model = Sequential()

for layer in base_model.layers:
    layer.trainable = False
    
inception_model.add(base_model)
inception_model.add(Flatten())
inception_model.add(Dense(1024,activation='relu'))
inception_model.add(Dense(3,activation='softmax'))


# In[5]:


inception_model.summary()


# ### Compile Model

# In[6]:


inception_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


# In[7]:


epochs = 10

history = inception_model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs
)


# In[8]:


fig = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','validation'])
plt.show()


# In[ ]:





# In[ ]:




