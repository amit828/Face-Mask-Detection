#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.callbacks import TensorBoard, ModelCheckpoint


# # DATA PREPROCESSING

# In[3]:


train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

#Train="C:\Users\Lenovo\face-mask-dataset\Dataset\train.zip"
#TRAINING_DIR = "./train"
training_set = train_datagen.flow_from_directory(r"C:\Users\Lenovo\New Masks Dataset\Train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[4]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r"C:\Users\Lenovo\New Masks Dataset\Test",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# # Initialising the cnn

# In[5]:


cnn = tf.keras.models.Sequential()


# # Convolution

# In[6]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# # Pooling

# In[7]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# # Adding a second convolutional layer

# In[8]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# # Flattening

# In[9]:


cnn.add(tf.keras.layers.Flatten())


# #  Full Connection

# In[10]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# # output layer

# In[11]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# # compiling the CNN

# In[12]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Training the cnn on the training set and evaluting it on test set

# In[13]:


cnn.fit(training_set, validation_data = test_set, epochs = 25)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




