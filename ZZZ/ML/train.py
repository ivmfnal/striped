import keras, numpy as np
from model import create_model
from keras.datasets import cifar10

model = create_model()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Convert class vectors to binary class matrices.
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train, x_test = x_train.astype(np.float32)/256, x_test.astype(np.float32)/256

batch_size = 1000

model.fit(x_train, y_train, batch_size=batch_size, verbose=1, validation_data = (x_test, y_test))


