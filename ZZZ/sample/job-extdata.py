from striped.job import SinglePointStripedSession as Session

from histbook import Hist, groupby
from histbook import bin as hbin
import numpy as np
import pandas as pd

worker_class = """
import numpy as np

class Worker(object):

    Columns = ["NJets"]

    def run(self, events, job, db):
	data = np.frombuffer(db["calib200"], "<f4")
        job.fill(x = data)
	job.message("average=%f" % (np.mean(data),))
"""

job_server = ("ifdb02.fnal.gov", 8765)
session = Session(job_server)

h = Hist(hbin("x", 20, 0, 1))

dataset = "Summer16.TTHH_TuneCUETP8M2T4_13TeV-madgraph-pythia8"
        

job = session.createJob(dataset, 
		    worker_class_source = worker_class, 
		    histograms = [h])
job.run()
runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
print "%s: %.6fM events, %.6fM events/second" % (dataset, float(nevents)/1000000, nevents/runtime/1000000)

data_frame = h.pandas()

print data_frame
