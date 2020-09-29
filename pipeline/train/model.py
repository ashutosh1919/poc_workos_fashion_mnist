import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import yaml
import pickle
import os
import sys


class LeNet(Sequential):
    """
    LeNet-5 model class, 2 convolution layers, 2 average pooling, 1 dense layer, activation = tanh
    """
    def __init__(self, input_shape=(28, 28, 1), nb_classes=10, optimizer="adam", metrics=["accuracy", "mse"]):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='tanh'))
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(nb_classes, activation='softmax'))

        self.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=metrics)