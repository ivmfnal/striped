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
                self.Columns = params["columns"]
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
                
        def preconvert_data(self, frame):
                x = frame.dot(self.Columns[0])
                y_ = frame.dot(self.Columns[1])
                n = len(x)
                return n, [x], [y]
            
                
        def frame(self, data):
            with self.Trace["model"]:
        
                with self.Trace["model/reset"]:
                    model = self.resetModel(self.Model)

                n, x, y_ = self.preconvert_data(data)

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

class ML_Keras_EvaluateWorker:

        def __init__(self, params, bulk, job, db):
                self.Bulk = bulk
                self.Columns = params["columns"]

                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)
                
                self.ModelConfig = params["model"]

                model = model_from_json(self.ModelConfig["config"])     
                #print  self.ModelConfig.keys(),   self.ModelConfig["loss"],   self.ModelConfig["metrics"]      
                model.compile(optimizer=optimizers.SGD(0.1, momentum=0.0), 
                        loss=self.ModelConfig["loss"], 
                        metrics=self.ModelConfig["metrics"])
                self.Model = model

                weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Weights0 = weights
                
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
        def getXY(self, frame):
                x = getattr(data, self.Columns[0])
                y_ = getattr(data, self.Columns[1])
                n = len(x)
                return n, [x], [y]

        def frame(self, data):
                model = self.Model
                model.set_weights(self.Weights0)
                
                n, x, y_ = self.getXY(data)

                #self.Job.message("run...")
                        
                loss, metric = model.test_on_batch(x, y_)
                            
                self.SumMetric += metric * n
                self.SumLoss += loss*n                        
                self.Samples += n
                
        def end(self):
                return {
                    "samples":  self.Samples,
                    "sumloss":  self.SumLoss,
                    "summetric":    self.SumMetric
                }
                
ML_Keras_EvaluateWorker_text = """
from striped.ml.keras_backend import ML_Keras_EvaluateWorker as Worker
from striped.ml import ML_EvaluateAccumulator as Accumulator
"""

ML_Keras_FitWorker_text = """
from striped.ml.keras_backend import ML_Keras_FitWorker as Worker
from striped.ml import ML_FitAccumulator as Accumulator
"""


