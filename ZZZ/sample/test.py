from striped.job import SinglePointStripedSession as Session

from histbook import Hist, groupby
from histbook import bin as hbin
import numpy as np
import pandas as pd

worker_class = """
import cloudpickle

class Worker(object):

    Columns = ["NJets"]

    def run(self, events, job):
        job.message("%d events" % (len(events),))
	x = 5/0
"""

session = Session(("ifdb01.fnal.gov", 8765))

h_by_dataset = Hist(hbin("NJets", 20, 0, 20), groupby("dataset"))

datasets = [
        "Summer16.TTHH_TuneCUETP8M2T4_13TeV-madgraph-pythia8"          		# 100000 events
]

class Callback:
	def on_message(self, wid, nevents, message):
		print "Message received from worker %d after seeing %d events: <%s>" % (wid, nevents, message)

	def on_exception(self, wid, info):
		print "Worker %d failed with exception:\n%s" % (wid, info)

callback = Callback()
        

for dataset_name in datasets:
    job = session.createJob(dataset_name, 
                            worker_class_source = worker_class, 
				user_callback = callback,
                            user_params = {
					"dataset":dataset_name,
					"data": np.arange(12).reshape((3,4))
				},
                            histograms = [h_by_dataset])
    job.run()
    runtime = job.TFinish - job.TStart
    nevents = job.EventsProcessed
    print "%s: %.6fM events, %.6fM events/second" % (dataset_name, float(nevents)/1000000, nevents/runtime/1000000)

data_frame = h_by_dataset.pandas()

print data_frame
