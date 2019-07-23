import keras, time
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from model_conv import create_model
from keras import backend as K
from keras.models import model_from_json, Model


config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
tf_session = tf.Session(config=config)
K.set_session(tf_session)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.asarray(x_train, dtype = np.float32)/255.0
x_test = np.asarray(x_test, dtype = np.float32)/255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


model = create_model()
cfg = model.to_json()
weights = model.get_weights()

m1 = model_from_json(cfg)  

opt = keras.optimizers.SGD(lr=0.2, decay=0.0001, momentum=0.8)

m1.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
m1.set_weights(weights)

for epoch in range(5):
    weights0 = m1.get_weights()
    m1.train_on_batch(x_train, y_train)
    weights1 = m1.get_weights()
    
    deltas = [w1-w0 for w1, w0 in zip(weights1, weights0)]
    for w, d in zip(weights0, deltas):
        w += d
    t0 = time.time()    
    m1.set_weights(weights0)
    print "set_weights:", time.time() - t0
    print m1.evaluate(x_test, y_test)
