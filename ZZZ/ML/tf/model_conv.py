import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D


def create_model():


    num_classes = 10
    model = Sequential()
    input_shape = (28, 28)
    model.add(Reshape((28,28,1), input_shape=input_shape))
    model.add(Conv2D(15, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(25, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(50, (3,3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(150, activation="tanh"))
    model.add(Dense(num_classes, activation="softmax"))

    return model
