#from __future__ import print_function
import os
from mnist_model import create_model
from striped.job import Session
import numpy as np

def pack_model(model, **params):
    data = {}
    data.update(params)
    data["model"] = {
        }
    return data

class MomentumOptimizer(object):

        def __init__(self, eta = 0.01, momentum = 0.9):
                self.Eta = eta
                self.Momentum = momentum
                self.PrevDeltas = None

        def __call__(self, params, grads):
                assert len(parms) == len(grads)
                deltas = [-eta*g for g in grads]
                prev_deltas = self.PrevDeltas
                if prev_deltas is not None and momentum != 0.0:
                        assert len(prev_deltas) == len(params)
                        for d, pd in zip(deltas, prev_deltas):
                                d += momentum * pd;
                params1 = [p + d for p, d in zip(params, deltas)]
                self.PrevDeltas = deltas
                return params1
                
class MLJob:

    def __init__(self, session, model, optimizer):
        self.Model = model
        self.Params = model.get_params()
        self.Grads = map(np.zeros_like, self.Params)
        self.NSamples = 0
        self.Session = session
        self.Runtime = None
        self.Optimizer = optimizer
        self.Sum = None

    def on_data(self, wid, nevents, data):
        if "sum" in data:
            self.NSamples += data["n"]
            if self.Sum is None:    self.Sum = data["sum"].copy()
            else:                   self.Sum += data["sum"]
            

    def run(self):
        job = self.Session.createJob("MNIST",
                            user_params = {
                                "model": {
                                        "config": model.config()
                                }
                            },
                            callbacks = [self],
                            worker_class_file="sumup_worker.py")
        job.run()
        self.Runtime = job.runtime

model = create_model()

session = Session("striped_dev.yaml")
optimizer = MomentumOptimizer()
job = MLJob(session, model, optimizer)
job.run()
print "NSamples:", job.NSamples
print "Average:", job.Sum/job.NSamples
print job.Runtime
