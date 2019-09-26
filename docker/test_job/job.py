
from striped.job import Session
from striped.job.callbacks import ProgressBarCallback
import numpy as np

class MyCallback:
	def __init__(self):
		self.N = 0

	def on_data(self, wid, nevents, data):
		#print ("on_data: %s %s %s" % (wid, nevents, data))
		self.N += data.get("count", 0)

cb = MyCallback()

dataset = "mnist"

session = Session("striped.yaml")

bulk_data = np.random.random((1000,1000))

job = session.createJob(dataset, 
				bulk_data = {"data":bulk_data},
				callbacks = [ProgressBarCallback(), cb],
				user_params = {"param":{"a":"b"}},
                            worker_class_file = "worker.py"
)
print ("running...")
job.run()
runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
print ("%s: %.6fM events, %.6fM events/second" % (dataset, float(nevents)/1000000, nevents/runtime/1000000))

print ("N=",cb.N)

