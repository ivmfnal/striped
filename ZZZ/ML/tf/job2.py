import time
from striped.job import Session

from striped.ml import MLSession
from model_conv import create_model

model = create_model()

session = Session("striped_dev.yaml")
ml_session = MLSession(session, model)



for t in range(5):
    for w in model.get_weights():
        print "weight:", w.dtype, w.shape, w.flat[:10]
    print "fit:     ", ml_session.fit("MNIST", "image", "labels",
            iterations=3, learning_rate=0.1, fraction=1.0, momentum=0.0)
    print "evaluate:", ml_session.evaluate("MNIST_test", "image", "labels")                            #, worker_file="evaluate_worker.py")


