import keras, numpy as np, json, time
from keras.models import model_from_json, Model
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD

class ML_Keras_FitWorker:

        def __init__(self, params, bulk, job, db):
                self.Job = job
                self.Bulk = bulk
                self.Params = params
                self.XColumn = params["xcolumn"]
                self.YColumn = params["ycolumn"]
                self.Weights0 = [p.copy() for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Model = self.initModel(self.Params, self.Weights0)
                
                self.Deltas = None
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
                
        def initModel(self, params, weights):
                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)
                
                self.ModelConfig = params["_model"]
                model = model_from_json(self.ModelConfig["config"])                
                
                #for w in self.Weights0:
                #    print "w: %s %s" % (w.dtype, w.shape)
                
                
                loss = self.ModelConfig.get("loss", "categorical_crossentropy")
                metric = self.ModelConfig.get("metric", "accuracy")
                
                optimizer_config = params.get("_optimizer", {})
                #self.Job.message("optimizer_config: %s" % (optimizer_config,))
                self.Iterations = optimizer_config.get("iterations", 1)
                self.MBSize = optimizer_config.get("mbsize", 20)
                optimizer = SGD(
                            lr =        optimizer_config.get("lr", 0.01), 
                            nesterov =  optimizer_config.get("nesterov", False), 
                            momentum =  optimizer_config.get("momentum", 0.0), 
                            decay =     optimizer_config.get("decay", 0.0001)
                )
                model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
                model.set_weights(weights)
                return model
                
        @property
        def Columns(self):
                return [self.XColumn, self.YColumn]

        def frame(self, data):
            with self.Trace["model"]:
        
                model = self.Model
                with self.Trace["model/set_weights"]:
                    model.set_weights(self.Weights0)

                x = getattr(data, self.XColumn)
                y_ = getattr(data, self.YColumn)
                n = len(x)


                #self.Job.message("run...")

                model = self.Model

                for t in range(self.Iterations):
                    with self.Trace["model/train"]:
                            history = model.fit(x, y_, batch_size=self.MBSize)
                            loss, metric = history.history["loss"][-1], history.history["acc"][-1]
                            
                with self.Trace["model/deltas"]:
                        weights1 = model.get_weights()
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
                return out



class ML_Keras_EvaluateWorker:

        def __init__(self, params, bulk, job, db):
                self.Bulk = bulk
                self.XColumn = params["xcolumn"]
                self.YColumn = params["ycolumn"]

                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)
                
                self.ModelConfig = params["_model"]

                model = model_from_json(self.ModelConfig["config"])                
                model.compile(optimizer=SGD(0.1, momentum=0.0), 
                        loss=self.ModelConfig["loss"], 
                        metrics=self.ModelConfig["metrics"])
                self.Model = model

                weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Weights0 = weights
                
                self.Samples = 0
                self.SumLoss = 0.0
                self.SumMetric = 0.0
                
        @property
        def Columns(self):
                return [self.XColumn, self.YColumn]

        def frame(self, data):
                model = self.Model
                model.set_weights(self.Weights0)
                
                x = getattr(data, self.XColumn)
                y_ = getattr(data, self.YColumn)
                n = len(x)


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


