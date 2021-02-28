
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

def getModel():
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(10,11,1)),
    tf.keras.layers.Conv2D(4, (1,1), activation='relu', input_shape=(10,11,1)),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
