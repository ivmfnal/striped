from striped.job import Session

from striped.ml import MLSession
from model_conv import create_model

model = create_model()

session = Session("striped_130tb.yaml")
ml_session = MLSession(session, model)


for t in range(5):
    print "fit:     ", ml_session.fit("MNIST", iterations=5, learning_rate=0.05, worker_file="fit_worker.py")
    print "evaluate:", ml_session.evaluate("MNIST_test")                            #, worker_file="evaluate_worker.py")


