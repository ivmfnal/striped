import keras, numpy as np, json, time
from keras.models import model_from_json, Model
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD

class ML_Keras_FitWorker:

        def __init__(self, params, bulk, xcolumn, ycolumn, optimizer = None):
                self.Bulk = bulk
                self.XColumn = xcolumn
                self.YColumn = ycolumn

                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)
                
                self.ModelConfig = params["_model"]
                model = model_from_json(self.ModelConfig["config"])                
                self.Weights0 = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                loss = self.ModelConfig.get("loss", "categorical_crossentropy")
                metric = self.ModelConfig.get("metric", "accuracy")
                
                optimizer_config = params.get("_optimizer", {})
                self.Iterations = optimizer_config.get("iterations", 1)
                if optimizer is None:
                    optimizer = SGD(
                                lr =        optimizer_config.get("lr", 0.01), 
                                nesterov =  optimizer_config.get("nesterov", False), 
                                momentum =  optimizer_config.get("momentum", 0.0), 
                                decay =     optimizer_config.get("decay", 0.0001)
                    )
                model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
                self.Model = model

                
                self.Deltas = map(np.zeros_like, weights)
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
                        
                model = self.Model

                for t in range(self.Iterations):
                    with self.Trace["train"]:
                            loss, metric = model.train_on_batch(x, y_)
                            
                with self.Trace["deltas"]:
                        weights1 = model.get_weights()
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

        def __init__(self, params, bulk, xcolumn, ycolumn):
                self.Bulk = bulk
                self.XColumn = xcolumn
                self.YColumn = ycolumn

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


