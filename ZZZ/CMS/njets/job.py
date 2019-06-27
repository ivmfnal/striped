import json
from QWorker import DistributedStripedSession as Session
from QWorker import Histogram

from datasets import Datasets

#from histbook import Hist, beside
#from histbook import bin as hbin
#import vegascope; canvas = vegascope.LocalCanvas()

registry_url = "http://ifdb01.fnal.gov:9867"
data_server_url = "http://dbweb7.fnal.gov:9091/striped/app"
    
session = Session(data_server_url, registry_url)

histograms_dump = {}

for dataset_name in Datasets:
    job = session.createJob(dataset_name, worker_class_file="worker.py", display=False)
    h = Histogram(0, 20, 20, title="N Jets")
    job.addDynamicHistogram(h, "njets", display=False)
    job.start()
    job.waitDone()
    runtime = job.TFinish - job.TStart
    nevents = job.EventsProcessed
    nworkers = len(job.WorkerAddresses)
    print "%s: %d events processed using %d workers in %.1f seconds, %.2e events/second" % (
                    dataset_name, nevents, nworkers, runtime, float(nevents)/runtime)

    histograms_dump[dataset_name] = json.loads(h.toJSON())
    
json.dump(histograms_dump, open("histograms.json", "w"),
        separators=(',',':'), indent=4, sort_keys=True)

