import keras, time
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from model_conv import create_model, digest
from keras import backend as K
from keras.models import model_from_json, Model
from tqdm import tqdm


config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
tf_session = tf.Session(config=config)
K.set_session(tf_session)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.asarray(x_train, dtype = np.float32)/256.0
x_test = np.asarray(x_test, dtype = np.float32)/256.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


lr = 0.01
frame_size = 1000
steps = 1
mbsize = 40
decay=0.0000
momentum=0.0

validate_frame_size = 500

model = create_model()
opt = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum)
#opt = keras.optimizers.adadelta(lr=lr, decay=decay)

model.compile(loss='categorical_crossentropy',
          optimizer=opt,
          metrics=['accuracy'])
          

weights0 = model.get_weights()



for epoch in range(1):
    deltas = map(np.zeros_like, weights0)
    ndeltas = 0
    for iframe, i0 in enumerate(range(0, len(x_train), frame_size)):
        if iframe in (31,30):
            x = x_train[i0:i0+frame_size]
            y = y_train[i0:i0+frame_size]
            mb = len(x)
            if mb:
                model.set_weights(weights0)
                print "x/y: %d %d %s/%s" % (iframe, len(x), digest(x), digest(y))
                print "model before:", digest(weights0)
                model.fit(x, y, batch_size=mbsize, verbose=False, shuffle=False)
                frame_deltas = [w1-w0 for w0, w1 in zip(weights0, model.get_weights())]
                print "frame deltas: %d: %s" % (iframe, digest(frame_deltas))
                for d, w, w0 in zip(deltas, model.get_weights(), weights0):
                    d += (w-w0)*mb
            ndeltas += mb
    for d in deltas:
        d /= ndeltas
    weights0 = [w0+d for w0, d in zip(weights0, deltas)]

    print "digest after :", digest(weights0)

    
    model.set_weights(weights0)
    for iframe, i0 in enumerate(range(0, len(x_test), validate_frame_size)):
        x = x_test[i0:i0+validate_frame_size]
        y = y_test[i0:i0+validate_frame_size]
        mb = len(x)
        if mb:
            model.set_weights(weights0)
            #print "x/y: %d %d %s/%s" % (iframe, len(x), np.mean(x*x), np.mean(y*y))
	    loss, accuracy = model.test_on_batch(x_test, y_test)
            model.fit(x, y, batch_size=mbsize, verbose=False, shuffle=False)
            frame_deltas = [w1-w0 for w0, w1 in zip(weights0, model.get_weights())]
            #print "frame deltas: %d: %s" % (iframe, [np.mean(d*d) for d in frame_deltas])
            for d, w, w0 in zip(deltas, model.get_weights(), weights0):
                d += (w-w0)*mb
    print "evaluate: loss=%f, accuracy=%.1f%%" % (loss, accuracy*100.0)
