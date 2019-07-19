from striped.ml.keras_backend import ML_Keras_EvaluateWorker
from striped.ml import ML_EvaluateAccumulator

class Worker(ML_Keras_EvaluateWorker):
        def __init__(self, params, bulk, job_interface, db_interface):
                ML_Keras_EvaluateWorker.__init__(self, params, bulk, "image", "labels")

class Accumulator(ML_EvaluateAccumulator):
        pass


