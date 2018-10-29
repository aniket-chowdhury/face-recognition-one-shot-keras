
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import tensorflow as tf


# In[3]:


import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Input
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l1,l2
from keras import backend as K
from keras.losses import binary_crossentropy,categorical_crossentropy


# In[4]:


from keras.layers import Lambda


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


input_shape = (200,200,1)
left = Input(input_shape)
right = Input(input_shape)


# In[26]:


try:
    model = Sequential()

    model.add(Conv2D(64,kernel_size=(20,20),input_shape=input_shape,padding='same',kernel_regularizer=l2(2e-5)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2,2),strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))

    model.add(Conv2D(64,kernel_size=(15,15),padding='same',kernel_regularizer=l2(2e-3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2,2),strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.40))

    model.add(Conv2D(128,kernel_size=(10,10),padding='same',kernel_regularizer=l2(2e-3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2,2),strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    model.add(Conv2D(128,kernel_size=(7,7),padding='same',kernel_regularizer=l2(1e-3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2,2),strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(256,kernel_size=(7,7),padding='same',kernel_regularizer=l2(1e-3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2,2),strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(18432//6,activation='sigmoid',kernel_regularizer=l2(1e-3)))

    left_output = model(left)
    right_output = model(right)

    layer_lambda = Lambda(lambda inputs:K.abs(inputs[0]-inputs[1]))
    layer_second_last = layer_lambda([left,right])

    out = Dense(1,activation='sigmoid') (layer_second_last)

    network = Model(inputs=[left,right],outputs=out)

    print(model.summary(),network.summary())
    
except Exception as e:
    print(e)

