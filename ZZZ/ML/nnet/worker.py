from striped.ml import ML_Accumulator
from striped.ml.nnet_backend import ML_NNet_Worker

class Worker(ML_NNet_Worker):

        def __init__(self, params, bulk, job_interface, db_interface):
                ML_NNet_Worker.__init__(self, params, bulk, "image", "labels")

class Accumulator(ML_Accumulator):
        pass

