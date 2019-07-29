import keras, numpy as np, json, time
from keras.models import model_from_json, Model
import tensorflow as tf
from keras import backend as K
from keras import optimizers

def digest(lst):
    if not isinstance(lst, list):
        lst = [lst]
    return [np.mean(w*w) for w in lst]

class ML_Keras_FitWorker:

        def __init__(self, params, bulk, job, db):
                self.Trace = None
                self.Job = job
                self.Bulk = bulk
                self.Params = params
                self.XColumn = params["xcolumn"]
                self.YColumn = params["ycolumn"]
                self.Iterations = params.get("iterations", 1)
                self.MBSize = params.get("mbsize", 40)
                self.ModelConfig = params["model"]
                self.Loss = self.ModelConfig.get("loss", "categorical_crossentropy")
                self.Metric = self.ModelConfig.get("metric", "accuracy")
                
                self.OpType = params["optimizer"]
                self.OptimizerParams = params["optimizer_params"]

                self.Deltas = None
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
                # Init Keras and Tensorflow
                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)

                self.Weights0 = [p.copy() for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Model = model_from_json(self.ModelConfig["config"])

        def resetModel(self, model):
                with self.Trace["model/reset/optimizer"]:
                    if self.OpType == "SGD":
                        optimizer = optimizers.SGD(**self.OptimizerParams)
                    elif self.OpType == "adadelta":
                        optimizer = optimizers.adadelta(**self.OptimizerParams)
                    elif self.OpType == "adagrad":
                        optimizer = optimizers.adagrad(**self.OptimizerParams)
                    else:
                        raise VaueError("Unknown optimizer type %s" % (self.OpType,))
                #self.Job.message("========= optimizer:%s, %s\n   mbsize=%d, iterations=%d" % (optimizer, optimizer_params, self.MBSize, self.Iterations))
                
                with self.Trace["model/reset/compile"]:
                    model.compile(optimizer=optimizer, loss=self.Loss, metrics=[self.Metric])
                with self.Trace["model/reset/set_weights"]:
                    model.set_weights(self.Weights0)
                
                return model
                
        @property
        def Columns(self):
                return [self.XColumn, self.YColumn]

        def frame(self, data):
            with self.Trace["model"]:
        
                with self.Trace["model/reset"]:
                    model = self.resetModel(self.Model)

                x = getattr(data, self.XColumn)
                y_ = getattr(data, self.YColumn)
                n = len(x)


                #self.Job.message("run...")

                #self.Job.message("mbsize: %s" % (self.MBSize,))
                #self.Job.message("initial_model: %s" % (digest(model.get_weights()),))
                #self.Job.message("x/y: %d %d %s/%s" % (data.rgid, len(x), np.mean(x*x), np.mean(y_*y_)))
                for t in range(self.Iterations):
                    with self.Trace["model/train"]:
                            history = model.fit(x, y_, batch_size=self.MBSize, verbose=False, shuffle=False)
                            loss, metric = history.history["loss"][-1], history.history["acc"][-1]
                            
                weights1 = model.get_weights()
                frame_deltas = [w1-w0 for w0, w1 in zip(self.Weights0, weights1)]
                #self.Job.message("frame_deltas: %d %s" % (data.rgid, digest(frame_deltas)))
                            
                with self.Trace["model/deltas"]:
                        if self.Deltas is None:
                            self.Deltas = [(w1 - w0)*n for w0, w1 in zip(self.Weights0, weights1)]
                        else:
                            for d, w1, w0 in zip(self.Deltas, weights1, self.Weights0):
                                d += (w1-w0)*n
                self.SumMetric += metric * n
                self.SumLoss += loss*n                        
                self.Samples += n
                
        def end(self):
                out = {"delta_%020d" % (i,): g for i, g in enumerate(self.Deltas)}
                out["samples"] = self.Samples
                out["sumloss"] = self.SumLoss
                out["summetric"] = self.SumMetric
                K.clear_session()
                return out


Worker = ML_Keras_FitWorker

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
        
Accumulator = ML_FitAccumulator
