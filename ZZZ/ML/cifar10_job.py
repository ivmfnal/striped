#from __future__ import print_function
import os
from model import create_model
from striped.job import Session
import numpy as np

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

def pack_model(model, **params):
    data = {}
    data.update(params)
    data["_model"] = {
        "config": model.to_json(),
        "weights": model.get_weights()
        }
    return data
    

class MLJob:

    def __init__(self, session, model):
        self.Model = model
        self.Deltas = map(np.zeros_like, model.get_weights())
        self.NSamples = 0
        self.Session = session
        self.Runtime = None

    def on_streams_updates(self, nevents, data):
        for k, lst in data:
                if k.startswith("dw"):
                        i = int(k[2:])
                        for v in lst:
                                self.Deltas += v

    def on_job_end(self, nsamples, error):
        if not error:
                for d in self.Deltas:
                        self.Deltas /= nsamples
                weights = [w+d for w, d in zip(self.Model.get_weights, self.Deltas)]
                self.Model.set_weights(weights)
        

    def run(self):
        job = self.Session.createJob("CIFAR-10",
                            user_params = {
                                "model": pack_model(self.Model, 
                                            loss = "categorical_crossentropy",
                                            lr = 0.001)
                                },
                            callbacks = [self],
                            worker_class_file="cifar10_worker.py")
        job.run()
        self.Runtime = job.runtime

model = create_model()

session = Session("striped.yaml")
job = MLJob(session, model)
job.run()
print job.Runtime
