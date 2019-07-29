import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D

import numpy as np

def create_model():

    np.random.seed(100)


    num_classes = 10
    model = Sequential()
    input_shape = (28, 28)
    model.add(Reshape((28,28,1), input_shape=input_shape))
    model.add(Conv2D(16, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(30, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(50, (3,3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(150, activation="tanh"))
    model.add(Dense(num_classes, activation="softmax"))

    return model

def digest(lst):
    if not isinstance(lst, list):
        lst = [lst]
    return [np.mean(w*w) for w in lst]
    
