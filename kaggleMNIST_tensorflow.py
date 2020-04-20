#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:09:02 2020

@author: adamepstein
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
data_dir = '/Users/adamepstein/Downloads/digit-recognizer'

train_df = pd.read_csv(data_dir + '/train.csv')
test_df = pd.read_csv(data_dir + '/test.csv')
#%%
Y_train = train_df["label"]
Y_train = to_categorical(Y_train)
X_train = train_df.loc[0:train_df.shape[0], "pixel0":"pixel783"]
X_test = test_df

numval = 8000
X_val = X_train[0:numval]
Y_val = Y_train[0:numval]
X_partial_train = X_train[numval:]
Y_partial_train = Y_train[numval:]
#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation = 'relu', input_shape = (784,)),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])

#%%
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(6, (3,3), activation = 'relu', input_shape = (28,28,1)),
#     tf.keras.layers.MaxPooling2D((2,2)),
#     tf.keras.layers.Conv2D(14,(3,3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D((2,2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation = 'relu'),
#     tf.keras.layers.Dense(10, activation = 'softmax')
#     ])


#%%
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',\
              metrics = ['accuracy'])

history = model.fit(X_partial_train, Y_partial_train , \
                    epochs = 100, batch_size = 64, \
                        validation_data = (X_val, Y_val))

#%%
model.summary()   
#%% 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, 'bo')
plt.plot(val_acc, 'b')
#%%
Y_predicted = model.predict(X_test)
#%%
predictions = np.zeros(28000)

for i in range(28000):
    predictions[i] = np.argmax(Y_predicted[i])


#%%    
# kaggleSubmission = pd.DataFrame(predictions)
kaggleSubmission = pd.DataFrame()
kaggleSubmission['ImageId'] = np.arange(1,28001)


# kaggleSubmission = test_df[['']]
kaggleSubmission['Label'] = np.uint8(predictions)
kaggleSubmission.to_csv('/Users/adamepstein/Downloads/digit-recognizer/Kaggle_submission.csv', index = False)

    
#I got a kaggle score of 0.94871    
    
    