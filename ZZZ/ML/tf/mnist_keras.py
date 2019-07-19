import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

from model_conv import create_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.asarray(x_train, dtype = np.float32)/255.0
x_test = np.asarray(x_test, dtype = np.float32)/255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


model = create_model()
opt = keras.optimizers.SGD(lr=0.2, decay=0.0001, momentum=0.8)

model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])

for epoch in range(5):
    weights0 = model.get_weights()
    model.train_on_batch(x_train, y_train)
    weights1 = model.get_weights()
    
    deltas = [w1-w0 for w1, w0 in zip(weights1, weights0)]
    for w, d in zip(weights0, deltas):
        w += d
        
    model.set_weights(weights0)
    print model.evaluate(x_test, y_test)
