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
            params["model"] = {
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
                deltas = [d/self.NSamples for d in self.Deltas]
                #print "deltas:", [np.mean(w*w) for w in deltas]
                weights = [w + d for w, d in zip(self.Model.get_weights(), deltas)]
                #weights = self.Optimizer(weights, self.Grads)
                self.Model.set_weights(weights)
                #for d in deltas:
                #    print "delta:", d.shape, d.flat[:10]

    DefaultOptimizer = {
        "type": "SGD",
        "lr":   0.01,
        "decay":    0.0
    }
                
    def run(self, dataset, xcolumn, ycolumn, iterations = 1, mbsize = 40, optimizer = None, optimizer_params = None, **args):
        optimizer = optimizer or "SGD"
        if optimizer_params is None:
            optimizer_params = self.DefaultOptimizer
        params, weights = self.pack_model()
        params["optimizer"] = optimizer
        params["optimizer_params"] = optimizer_params
        params["iterations"] = iterations
        params["mbsize"] = mbsize
        params["xcolumn"] = xcolumn
        params["ycolumn"] = ycolumn
        #print "ML_FitJob: worker_file: %s, worker_text: [%s]" % (self.WorkerFile, self.WorkerText[:50] if self.WorkerText is not None else None)
        job = self.Session.createJob(dataset,
                            user_params = params,
                            bulk_data = weights,
                            callbacks = [self],
                            worker_class_file=self.WorkerFile,
                            worker_class_text=self.WorkerText,
                            **args)
        job.run()
        self.Runtime = job.runtime
        
class ML_EvaluateJob:

    def __init__(self, session, model, worker_file=None, worker_text=None, 
            loss="categorical_crossentropy",
            metric = "accuracy"):
        self.Model = model
        self.Deltas = map(np.zeros_like, model.get_weights())
        self.Session = session
        self.Runtime = None
        self.SumLoss = 0.0
        self.WorkerText = worker_text
        self.WorkerFile = worker_file
        self.SumMetric = 0.0
        self.NSamples = 0
        self.Loss = loss
        self.Metric = metric

    def pack_model(self):
            params = {}
            model = self.Model
            try:        
                cfg = model.to_json()
            except:
                cfg = model.config()
            params["model"] = {
                "config": cfg,
                "loss": self.Loss,
                "metrics":      [self.Metric]
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
                
    def run(self, dataset, xcolumn, ycolumn, **args):
        params, weights = self.pack_model()
        params["xcolumn"] = xcolumn
        params["ycolumn"] = ycolumn
        job = self.Session.createJob(dataset,
                            user_params = params,
                            bulk_data = weights,
                            callbacks = [self],
                            worker_class_file=self.WorkerFile,
                            worker_class_text=self.WorkerText,
                            **args)
        job.run()
        self.Runtime = job.runtime
        

class MLSession(object):
    def __init__(self, striped_session, model, platform="keras"):
        self.Model = model
        self.StripedSession = striped_session
        self.Platform = platform
        
        
    def fit(self, dataset, xcolumn, ycolumn, iterations=1, worker_file=None, worker_text=None, optimizer=None, optimizer_params=None, **args):
        if worker_file is None and worker_text is None:
            if self.Platform == "keras":
                from .keras_backend import ML_Keras_FitWorker_text
                worker_text = ML_Keras_FitWorker_text
            else:
                raise ValueError("Unknown ML platform %s" % (self.Platform,))
        job = ML_FitJob(self.StripedSession, self.Model, worker_file=worker_file, worker_text=worker_text)
        job.run(dataset, xcolumn, ycolumn, iterations=iterations, optimizer=optimizer, optimizer_params = optimizer_params, **args)
        return job.Loss, job.Metric
        
    def evaluate(self, dataset, xcolumn, ycolumn, worker_file=None, worker_text=None, **args):
        if worker_file is None and worker_text is None:
            if self.Platform == "keras":
                from .keras_backend import ML_Keras_EvaluateWorker_text
                worker_text = ML_Keras_EvaluateWorker_text
                #print "MLSession: worker_text: [%s]" % (worker_text,)
            else:
                raise ValueError("Unknown ML platform %s" % (self.Platform,))
        job = ML_EvaluateJob(self.StripedSession, self.Model, worker_file=worker_file, worker_text=worker_text)
        job.run(dataset, xcolumn, ycolumn, **args)
        return job.Loss, job.Metric
        
        
        
