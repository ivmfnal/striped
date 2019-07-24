import time
from striped.job import Session

from striped.ml import MLSession
from model_conv import create_model

model = create_model()

session = Session("striped_130tb.yaml")
ml_session = MLSession(session, model)



for t in range(10):
    #for w in model.get_weights():
    #    print "weight:", w.dtype, w.shape, w.flat[:10]
    print "fit:     ", ml_session.fit("MNIST", "image", "labels",
            learning_rate=0.01, fraction=1.0, momentum=0.5, decay = 0.0001)
    loss, accuracy = ml_session.evaluate("MNIST_test", "image", "labels")
    print "evaluate: loss: %.3f, accuracy: %.1f%%" % (loss, accuracy*100)


