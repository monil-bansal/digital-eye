#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pyswarms as ps
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,make_scorer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt
from gtts import gTTS
import os
import playsound


# In[4]:





# In[6]:


class model:
    def __init__(self):
        a = 5
        b = 6
        c = 7
        
    def train(self,X_train,y_train,X_test,y_test,clf,params):
        model = clf()
        layer_1 = Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
        layer_2 = MaxPooling2D(pool_size=2)
        layer_3 = Conv2D(32, kernel_size=3, activation='relu')
        layer_4 = MaxPooling2D(pool_size=2)
        layer_5 = Dropout(0.5)
        layer_6 = Flatten()
        layer_7 = Dense(128, activation="relu")
        layer_8 = Dropout(0.5)
        layer_9 = Dense(10, activation='softmax')

        ## Add the layers to the model
        model.add(layer_1)
        model.add(layer_2)
        model.add(layer_3)
        model.add(layer_4)
        model.add(layer_5)
        model.add(layer_6)
        model.add(layer_7)
        model.add(layer_8)
        model.add(layer_9)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

        return model


# In[ ]:




