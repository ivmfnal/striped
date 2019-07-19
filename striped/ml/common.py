import numpy as np

class ML_FitAccumulator:

    def __init__(self, params, bulk, job_interface, db_interface):
                #weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Deltas = None
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
    def add(self, data):
        deltas = [p for n, p in sorted(data.items()) if n.startswith("delta_")]
        if self.Deltas is None:
            self.Deltas = [d.copy() for d in deltas]
        else:
            for d, dd in zip(self.Deltas, deltas):
                d += dd
        self.SumLoss += data["sumloss"]
        self.Samples += data["samples"]
        self.SumMetric += data["summetric"]
        
    def values(self):
        #print "ML_Accumulator.values(): sending grads"
        return {"deltas": self.Deltas, "samples":self.Samples, 
                "sumloss": self.SumLoss, "summetric":self.SumMetric
                }
        
class ML_EvaluateAccumulator:

    def __init__(self, params, bulk, job_interface, db_interface):
                #weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
    def add(self, data):
        self.SumLoss += data["sumloss"]
        self.Samples += data["samples"]
        self.SumMetric += data["summetric"]
        
    def values(self):
        return {"samples":self.Samples, 
                "sumloss": self.SumLoss, "summetric":self.SumMetric
                }
        
class MomentumOptimizer:
    def __init__(self, lr, momentum = 0.0):
        self.LR = lr
        self.Momentum = momentum
        self.Deltas = None
        
    def apply(self, weights, grads):
        deltas = [-g*self.LR for g in grads]
        if self.Deltas is not None and self.Momentum != 0.0:
            for d, dd in zip(deltas, self.Deltas):
                deltas += self.Momentum * dd
        if self.Momentum != 0.0:
            self.Deltas = deltas
        return [w+d for w, d in zip(weights, deltas)]
        
    __call__ = apply
    
class ML_FitJob:

    def __init__(self, session, model, loss="categorical_crossentropy",
            metric = "accuracy",
            worker_file=None, worker_text=None, optimizer = None):
        self.Model = model
        self.Deltas = map(np.zeros_like, model.get_weights())
        self.NSamples = 0
        self.Session = session
        self.Runtime = None
        self.SumLoss = 0.0
        self.Loss = self.Metric = None
        self.WorkerText = worker_text
        self.WorkerFile = worker_file
        self.SumMetric = 0.0
        self.Metric = metric
        self.Loss = loss

    def pack_model(self):
            params = {}
            model = self.Model
            try:        
                cfg = model.to_json()
            except:
                cfg = model.config()
            params["_model"] = {
                "config": cfg,
                "loss": self.Loss,
                "metrics":      [self.Metric]
                }
            weights = {"weight_%020d" % (i,):w for i, w in enumerate(model.get_weights())}
            return params, weights
    
    def on_data(self, wid, nevents, data):
        #print "ML_FitJob.on_data():", data
        for g, gg in zip(self.Deltas, data["deltas"]):
            g += gg
        self.NSamples += data["samples"]
        self.SumLoss += data["sumloss"]
        self.SumMetric += data["summetric"]
        
    def on_job_finish(self, nsamples, error):
        if not error:
                self.Metric = self.SumMetric / self.NSamples
                self.Loss = self.SumLoss / self.NSamples
                weights = [w + d/self.NSamples for w, d in zip(self.Model.get_weights(), self.Deltas)]
                #weights = self.Optimizer(weights, self.Grads)
                self.Model.set_weights(weights)
                
    def run(self, dataset, learning_rate = 0.01, iterations = 1, nesterov = False, momentum = 0.0, **args):
        params, weights = self.pack_model()
        params["_optimizer"] = {
            "type":             "SGD",
            "lr":               learning_rate,
            "iterations":       iterations,
            "nesterov":         nesterov,
            "momentum":         momentum
        }
        job = self.Session.createJob(dataset,
                            user_params = params,
                            bulk_data = weights,
                            callbacks = [self],
                            worker_class_file=self.WorkerFile,
                            **args)
        job.run()
        self.Runtime = job.runtime
        
class ML_EvaluateJob:

    def __init__(self, session, model, worker_file=None, worker_text=None, optimizer = None):
        self.Model = model
        self.Deltas = map(np.zeros_like, model.get_weights())
        self.Session = session
        self.Runtime = None
        self.SumLoss = 0.0
        self.WorkerText = worker_text
        self.WorkerFile = worker_file
        self.SumMetric = 0.0
        self.NSamples = 0
        self.Loss = self.Metric = None

    def pack_model(self):
            params = {}
            model = self.Model
            try:        
                cfg = model.to_json()
            except:
                cfg = model.config()
            params["_model"] = {
                "config": cfg,
                "loss": 'categorical_crossentropy',
                "metrics":      ["accuracy"]
                }
            weights = {"weight_%020d" % (i,):w for i, w in enumerate(model.get_weights())}
            return params, weights
    
    def on_data(self, wid, nevents, data):
        #print "ML_EvaluateJob.on_data(): data:", data
        self.NSamples += data["samples"]
        self.SumLoss += data["sumloss"]
        self.SumMetric += data["summetric"]
        
    def on_job_finish(self, nsamples, error):
        if not error:
                self.Metric = self.SumMetric / self.NSamples
                self.Loss = self.SumLoss / self.NSamples
                
    def run(self, dataset, **args):
        params, weights = self.pack_model()
        job = self.Session.createJob(dataset,
                            user_params = params,
                            bulk_data = weights,
                            callbacks = [self],
                            worker_class_file=self.WorkerFile,
                            **args)
        job.run()
        self.Runtime = job.runtime
        

class MLSession(object):
    def __init__(self, striped_session, model, platform="keras"):
        self.Model = model
        self.StripedSession = striped_session
        self.Platform = platform
        
        
    def fit(self, dataset, iterations=1, learning_rate=0.01, worker_file=None, worker_text=None, **args):
        if worker_file is None and worker_text is None:
            if self.Platform == "keras":
                from .keras_backend import ML_Keras_FitWorker_text
                worker_text = ML_Keras_FitWorker_text
            else:
                raise ValueError("Unknown ML platform %s" % (self.Platform,))
        job = ML_FitJob(self.StripedSession, self.Model, worker_file=worker_file, worker_text=worker_text)
        job.run(dataset, learning_rate=learning_rate, iterations=iterations, **args)
        return job.Loss, job.Metric
        
    def evaluate(self, dataset, worker_file=None, worker_text=None, **args):
        if worker_file is None and worker_text is None:
            if self.Platform == "keras":
                from .keras_backend import ML_Keras_EvaluateWorker_text
                worker_text = ML_Keras_EvaluateWorker_text
            else:
                raise ValueError("Unknown ML platform %s" % (self.Platform,))
        job = ML_EvaluateJob(self.StripedSession, self.Model, worker_file=worker_file, worker_text=worker_text)
        job.run(dataset, **args)
        return job.Loss, job.Metric
        
        
        
