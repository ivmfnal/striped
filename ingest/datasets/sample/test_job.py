from striped.job import SinglePointStripedSession as Session

from histbook import Hist, groupby
from histbook import bin as hbin
import numpy as np
import pandas as pd

worker_class = """
import time, socket

class Worker(object):

    Columns = ["A.p", "n"]

    def run(self, events, job, database):
        job.fill(n = events.n)
        job.fill(p = events.A.p)
"""

dataset = "Sample"

session = Session("striped_dev.yaml")

h_n = Hist(hbin("n", 20, 0, 20))
h_p = Hist(hbin("p", 20, 0, 20))

job = session.createJob(dataset, 
			user_params={"param":{"hello":"world"}},
                            worker_class_source = worker_class, 
                            histograms = [h_n, h_p])
job.run()
runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
print "%s: %.6fM events, %.6fM events/second" % (dataset, float(nevents)/1000000, nevents/runtime/1000000)

print h_n.pandas()
print h_p.pandas()



