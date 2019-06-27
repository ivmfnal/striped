from striped.job import SinglePointStripedSession as Session
import numpy as np, time

worker_class = """

import time

class Worker(object):

    Columns = ["image", "labels"]

    def __init__(self, params, bulk, job, db):
	self.NEvents = 0
	self.NFrames = 0

    def run(self, events):
	self.NFrames += 1
	self.NEvents += len(events)
	#time.sleep(1)
	#return {"Events_in_frame":len(events)}

    def end(self):
	return {"NEvents":self.NEvents, "NFrames":self.NFrames}
	pass

class Accumulator(object):

	def __init__(self, params, bulk, job, db):
		self.NEvents = 0
		self.NFrames = 0

	def accumulate(self, nevents, data):
		self.NEvents += data.get("NEvents", 0)
		self.NFrames += data.get("NFrames", 0)
		return {"events_processed":self.NEvents}

	def accumulated(self, nevents):
		return {"NEvents":self.NEvents, "NFrames":self.NFrames}
		
"""

session = Session("striped_dev.yaml")

bulk = np.random.random((10,10))

class callback:

	def on_data(self, wid, n, data):
		print n, data

dataset_name = "MNIST"
job = session.createJob(dataset_name,
		    worker_class_source = worker_class, 
		    #bulk_data = {"x":bulk },
		    callbacks = [callback()],
		    user_params = {
				"dataset":dataset_name
			})
job.run()
runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
print "%s: %.6fM events, %.6fM events/second" % (dataset_name, float(nevents)/1000000, nevents/runtime/1000000)

