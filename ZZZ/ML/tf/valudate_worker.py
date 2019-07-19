from striped.ml.keras_backend import ML_Keras_Worker
from striped.ml import ML_Accumulator

class Worker(ML_Keras_ValidateWorker):
        def __init__(self, params, bulk, job_interface, db_interface):
                ML_Keras_Worker.__init__(self, params, bulk, "image", "labels", 
                        optimizer = SGD(lr=0.1, nesterov=False, momentum=0.5))

class Accumulator(ML_TrainAccumulator):
        pass


