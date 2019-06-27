from striped.ml.keras_backend import ML_Keras_Worker
from striped.ml import ML_Accumulator

class Worker(ML_Keras_Worker):

        def __init__(self, params, bulk, job_interface, db_interface):
		ML_Keras_Worker.__init__(self, params, bulk, "image", "labels")

class Accumulator(ML_Accumulator):
	pass

