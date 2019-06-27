import numpy as np

class ML_Accumulator:

    def __init__(self, params, bulk, job_interface, db_interface):
                weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Grads = map(np.zeros_like, weights)
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
    def add(self, data):
        grads = [p for n, p in sorted(data.items()) if n.startswith("grad_")]
        for s, d in zip(self.Grads, grads):
            s += d
        self.SumLoss += data["sumloss"]
        self.Samples += data["samples"]
        self.SumMetric += data["summetric"]
        
    def values(self):
        #print "ML_Accumulator.values(): sending grads"
        return {"grads": self.Grads, "samples":self.Samples, 
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
    
class ML_Job:

    class SimpleOptimizer:
    
        def __init__(self, lr, momentum = 0.0):
            self.LR = lr
            self.Deltas = None
            

    def __init__(self, session, model, worker_file=None, worker_text=None, optimizer = None, **args):
        self.Model = model
        self.Grads = map(np.zeros_like, model.get_weights())
        self.NSamples = 0
        self.Session = session
        self.Runtime = None
        self.SumLoss = 0.0
        self.Loss = None
        self.WorkerText = worker_text
        self.WorkerFile = worker_file
        self.Optimizer = optimizer or MomentumOptimizer(0.001)
        self.Args = args
        self.SumMetric = 0.0

    def pack_model(self):
            data = {}
            model = self.Model
            try:        
                cfg = model.to_json()
            except:
                cfg = model.config()
            data["_model"] = {
                "config": cfg,
                "loss": 'categorical_crossentropy',
                "metrics":      ["accuracy"]
                }
            weights = {"weight_%020d" % (i,):w for i, w in enumerate(model.get_weights())}
            return data, weights
    
    def on_data(self, wid, nevents, data):
        #print "ML_Job.on_data(): keys:", data.keys()
        for g, gg in zip(self.Grads, data["grads"]):
            g += gg
        self.NSamples += data["samples"]
        self.SumLoss += data["sumloss"]
        self.SumMetric += data["summetric"]
        
    def on_job_finish(self, nsamples, error):
        if not error:
                for g in self.Grads:
                        g /= self.NSamples
                self.Metric = self.SumMetric / self.NSamples
                self.Loss = self.SumLoss / self.NSamples
        weights = self.Model.get_weights()
        weights = self.Optimizer(weights, self.Grads)
        self.Model.set_weights(weights)
                
    def run(self, dataset, learning_rate = 0.5):
        params, weights = self.pack_model()
        params["lr"] = learning_rate
        job = self.Session.createJob(dataset,
                            user_params = params,
                            bulk_data = weights,
                            callbacks = [self],
                            worker_class_file=self.WorkerFile,
                            **self.Args)
        job.run()
        self.Runtime = job.runtime
