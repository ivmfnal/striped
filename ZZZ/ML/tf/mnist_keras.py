import keras, time
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from model_conv import create_model
from keras import backend as K
from keras.models import model_from_json, Model
from tqdm import tqdm


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


lr = 0.01
n = 1000
steps = 1
mbsize = 40

model = create_model()
opt = keras.optimizers.SGD(lr=lr, decay=0.0001, momentum=0.5)

model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
          

weights0 = model.get_weights()
for epoch in range(50):
    deltas = map(np.zeros_like, weights0)
    for i in tqdm(range(0, len(x_train), n)):
        x = x_train[i:i+n]
        y = y_train[i:i+n]
        mb = len(x)
        if mb:
            model.set_weights(weights0)
            model.fit(x, y, batch_size=mbsize, verbose=False)
            for d, w, w0 in zip(deltas, model.get_weights(), weights0):
                d += (w-w0)*mb
    weights0 = [w0+d/len(x_train) for w0, d in zip(weights0, deltas)]
    
    model.set_weights(weights0)
    loss, accuracy = model.test_on_batch(x_test, y_test)
    print "evaluate: loss=%f, accuracy=%.1f%%" % (loss, accuracy*100.0)
