import json
from QWorker import DistributedStripedSession as Session
from QWorker import Histogram

from datasets import Datasets

from histbook import Hist, beside
from histbook import bin
#import vegascope; canvas = vegascope.LocalCanvas()

registry_url = "http://ifdb01.fnal.gov:9867"
data_server_url = "http://dbweb7.fnal.gov:9091/striped/app"
    
session = Session(data_server_url, registry_url)

muon_e = Hist(
    bin("sqrt(mu_px**2+mu_py**2+mu_pz**2)", 10, 10.0, 1000.0),
    bin("sqrt(mu_e**2-mu_px**2-mu_py**2-mu_pz**2)", 15, 0.1, 0.15)
)

job = session.createJob("Summer16.SMS-T5qqqqZH-mGluino1700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",
    worker_class_file="worker_new.py", display=False)
job.addHistogram(muon_e, ["mu_e","mu_px","mu_py","mu_pz"])

job.start()
job.waitDone()

print muon_e.pandas()
