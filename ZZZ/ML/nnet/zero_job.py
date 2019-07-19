from striped.job import Session

from striped.ml import ML_Job
from model import create_model

model = create_model()

session = Session("striped_130tb.yaml")

for epoch in range(5):
	job = ML_Job(session, model, worker_file="worker.py")
	job.run("MNIST", 0.1)
	print "epoch: %d, runtime: %f, loss: %s" % (epoch+1, job.Runtime, job.Loss)
