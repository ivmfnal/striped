from striped.job import SinglePointStripedSession as Session
import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import sys, time, getopt

from striped.common import Tracer
T = Tracer()

Usage = """
python verify.py
"""

opts, args = getopt.getopt(sys.argv[1:], "?h")
opts = dict(opts)

if "-?" in opts or "-h" in opts:
    print Usage
    sys.exit(1)

#job_server_address = ("dbwebdev.fnal.gov", 8765) 
job_server_address = ("ifdb01.fnal.gov", 8765) 
session = Session(job_server_address)

class Callback:

    def on_streams_update(self, nevents, data):
        if "message" in data:
            for msg in data["message"]:
                print msg

job = session.createJob("Bliss", 
		    use_data_cache = False, 
                    user_callback = Callback(),
                    worker_class_file="verify_worker.py")

job.run()


