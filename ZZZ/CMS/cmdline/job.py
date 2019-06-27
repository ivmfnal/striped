import matplotlib.pyplot as plt

from QWorker import DistributedStripedSession as Session
from QWorker import Histogram

#from histbook import Hist, beside
#from histbook import bin as hbin
#import vegascope; canvas = vegascope.LocalCanvas()

registry_url = "http://ifdb01.fnal.gov:9867"
data_server_url = "http://dbweb7.fnal.gov:9091/striped/app"
    
session = Session(data_server_url, registry_url)

#hnmu = Hist(hbin("nmu", 5, 0, 5))
#h_e_plus_e = Hist(hbin("e1+e2", 20, 0, 1000))
#h_e_minus_e = Hist(hbin("e1-e2", 20, -1000, 1000))
#h_mass = Hist(hbin("mass", 9, 10, 100))

h_mu_mu_mass = Histogram(10, 100.0, 90, title="2-Muon mass")

dataset_name = "Summer16.ZJetsToNuNu_HT-100To200_13TeV-madgraph"
job = session.createJob(dataset_name, worker_class_file="worker.py", fraction=0.5)

job.addDynamicHistogram(h_mu_mu_mass, "mu_mu_mass")
#job.addHistogram(hnmu, "nmu")
#job.addHistogram(h_e_plus_e, "e1", "e2")
#job.addHistogram(h_e_minus_e, "e1", "e2")
#job.addHistogram(h_mass, "mass")
job.start()
job.waitDone()


runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
nworkers = len(job.WorkerAddresses)
print "Job finished. %d events processed using %d workers in %.1f seconds, %.2e events/second" % (
                nevents, nworkers, runtime, float(nevents)/runtime)

print h_mu_mu_mass.toJSON()


#beside(hnmu.step("nmu"),
#    h_mass.step("mass"),
#    h_e_plus_e.step("e1+e2"),
#    h_e_minus_e.step("e1-e2")
#      ).to(canvas)


