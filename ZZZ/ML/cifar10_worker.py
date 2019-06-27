import keras, numpy as np, json, time
from keras.models import model_from_json, Model
#import tensorflow as tf
#from keras import backend as K

class Worker:

        Columns = ["x", "y"]

        def __init__(self):
                self.Model = None
                self.Weights0 = None
                self.Session = None

        def unpack_model(self, params):
            model = model_from_json(params["_model"]["config"])
            optimizer = keras.optimizers.SGD(lr=params["lr"], nesterov=True)
            model.compile(optimizer=optimizer, loss=params["loss"])
            return model, params["_model"]["weights"]

        def run(self, data, job, db):
                job.message("run...")
                print "run() entry"
                t0 = time.time()
                if False and self.Session is None:
                        print "initializing session..."
                        config = tf.ConfigProto()
                        config.intra_op_parallelism_threads = 5
                        config.inter_op_parallelism_threads = 5
                        self.Session = tf.Session(config=config)
                        K.set_session(self.Session)
                        print "session initialized"
                        
                if self.Model is None:
                        print "creating model..."
                        with self.Trace["create_model"]:
                                model, weights = self.unpack_model(job["model"])
                                self.Model = model
                                self.Weights0 = weights
                        print "model created"
            
                self.Model.set_weights(self.Weights0)

                model = self.Model
                
                x = data.x
                y = data.y

                with self.Trace["train"]:
                        print "train"
                        model.train_on_batch(x, y)
                        print "train done"

                with self.Trace["deltas"]:
                        print "sending"
                        weights1 = model.get_weights()
                        for i, (w0, w1) in enumerate(zip(self.Weights0, weights1)):
                                d = (w1 - w0) * len(x)
                                job.send("dw%d" % (i,), d)
                        print "sending done"

                job.message("time=%f" % (time.time() - t0))

                
                

        

        
