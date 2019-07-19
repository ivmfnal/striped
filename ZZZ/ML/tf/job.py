from striped.job import Session

from striped.ml import ML_Job, MomentumOptimizer
from model import create_model

model = create_model()

session = Session("striped_dev.yaml")

for epoch in range(5):
	job = ML_Job(session, model, worker_file="worker.py", fraction=1.0)
	job.run("MNIST", iterations=5)
	print "epoch: %d, runtime: %f, loss: %s, accuracy: %.1f%%" % (epoch+1, job.Runtime, job.Loss, job.Metric*100)
