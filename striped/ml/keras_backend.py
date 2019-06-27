import keras, numpy as np, json, time
from keras.models import model_from_json, Model
import tensorflow as tf
from keras import backend as K

class ML_Keras_Worker:

        def __init__(self, params, bulk, xcolumn, ycolumn):
                self.Bulk = bulk
                self.XColumn = xcolumn
                self.YColumn = ycolumn

                config = tf.ConfigProto()
                config.intra_op_parallelism_threads = 1
                config.inter_op_parallelism_threads = 1
                tf_session = tf.Session(config=config)
                K.set_session(tf_session)

                model = model_from_json(params["_model"]["config"])
                optimizer = keras.optimizers.SGD(lr=1.0, nesterov=False, momentum=0.0)
                model.compile(optimizer=optimizer, loss=params["_model"]["loss"], metrics=params["_model"]["metrics"])
                self.Model = model

                weights = [p for n, p in sorted(bulk.items()) if n.startswith("weight_")]
                self.Weights0 = weights
                
                self.Grads = map(np.zeros_like, weights)
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

                with self.Trace["evaluate"]:
                        ret = model.evaluate(x, y_, batch_size=n, verbose=0)
                        loss = ret[0]
                        metrics = ret[1:]
                        #self.Job.message("evaluate() -> %s" % (ret,))
                        self.SumLoss += loss*n
                
                with self.Trace["train"]:
                        loss, metric = model.train_on_batch(x, y_)
                        self.SumMetric += metric * n
                        
                with self.Trace["deltas"]:
                        weights1 = model.get_weights()
                        for g, w1, w0 in zip(self.Grads, weights1, self.Weights0):
                            g -= (w1-w0)*n
                self.Samples += n
                
        def end(self):
                out = {"grad_%020d" % (i,): g for i, g in enumerate(self.Grads)}
                out["samples"] = self.Samples
                out["sumloss"] = self.SumLoss
                out["summetric"] = self.SumMetric
                return out
