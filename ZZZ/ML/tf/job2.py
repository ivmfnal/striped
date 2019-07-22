from striped.job import Session

from striped.ml import MLSession
from model_conv import create_model

model = create_model()

session = Session("striped_dev.yaml")
ml_session = MLSession(session, model)


for t in range(5):
    print "fit:     ", ml_session.fit("MNIST", "image", "labels",
            iterations=5, learning_rate=0.1, fraction=0.1, momentum=0.9)
    print "evaluate:", ml_session.evaluate("MNIST_test", "image", "labels")                            #, worker_file="evaluate_worker.py")


