from striped.job import SinglePointStripedSession as Session
import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import sys, time, getopt

from striped.common import Tracer
T = Tracer()

Usage = """
python all_hpix.py
"""

opts, args = getopt.getopt(sys.argv[1:], "?h")
opts = dict(opts)

if "-?" in opts or "-h" in opts:
    print Usage
    sys.exit(1)

#job_server_address = ("dbwebdev.fnal.gov", 8765) 
job_server_address = ("ifdb01.fnal.gov", 8765) 
session = Session(job_server_address)

class Collector:

    def __init__(self):
        self.Collected = []

    def on_streams_update(self, nevents, data):
        if "message" in data:
            for msg in data["message"]:
                print msg
        if "hpix" in data:
            self.Collected.extend(data["hpix"])
            
    def hpix(self):
        hpix_set = set()
        for arr in self.Collected:
            hpix_set |= set(list(arr))
        return sorted(list(hpix_set))
        
collector = Collector()        
job = session.createJob("Bliss", 
		            use_data_cache = True, 
                    user_callback = collector,
                    worker_class_file="all_hpix_worker.py")

job.run()

hpix = collector.hpix()
print len(hpix), hpix[0], hpix[-1]

