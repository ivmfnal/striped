import os
from striped.job import Session
import numpy as np


session = Session("striped_130tb.yaml")
dataset = "NanoTuples-2016_QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8"

class DataCallback:

    def __init__(self):
        self.Sum_pt = 0.0
        self.N = 0
        self.AveragePt = None
        
    def on_data(self, wid, nevents, data):
        self.N += data["n_jets"]
        self.Sum_pt += data["sum_pt"]
    
    def on_job_finish(self, nsamples, error):
        self.AveragePt = self.Sum_pt/self.N

data_collector = DataCallback()

job = session.createJob(dataset, 
        user_params = {"dataset":dataset},
        bulk_data = {"calibrations":np.random.random((100,100))},  
        callbacks = [data_collector],
        worker_class_file="nano_worker.py")
job.run()
print "Jets: %d, average pt: %f" % (data_collector.N, data_collector.AveragePt)

