import time
from striped.job import Session

from striped.ml import MLSession
from model_conv import create_model, digest

model = create_model()

session = Session("striped_dev.yaml")
ml_session = MLSession(session, model)

optimizer_params = {            # SGD
    "lr":           0.01,
    "momentum":     0.0,
    "decay":        0.0000
}

#optimizer_params = {            # adadelta
#    "lr":           1.0,
#    "decay":        0.0001
#}

optimizer = "SGD"

for t in range(10):
    #for w in model.get_weights():
    #    print "weight:", w.dtype, w.shape, w.flat[:10]
    
    t0 = time.time()
    
    print "digest before:", digest(model.get_weights())
    
    loss, accuracy = ml_session.fit("MNIST", "image", "labels", fraction=1.0,
            optimizer = optimizer,
            optimizer_params = optimizer_params
            )
    
    
    """
    loss, accuracy = ml_session.fit("MNIST", "image", "labels", fraction=1.0,
            worker_file = "keras_fit.py",
            optimizer = optimizer,
            optimizer_params = optimizer_params
            )
    """
    
    
    print "digest after :", digest(model.get_weights())


    t = time.time() - t0
    print "fit:      loss: %.3f, accuracy: %.1f%%, time: %.1f" % (loss, accuracy*100, t)
    
    #print "weights:", weights_digest(model)
            
    loss, accuracy = ml_session.evaluate("MNIST_test", "image", "labels")
    print "evaluate: loss: %.3f, accuracy: %.1f%%" % (loss, accuracy*100)


