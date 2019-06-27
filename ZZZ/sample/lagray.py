from striped.job import SinglePointStripedSession as Session

from histbook import Hist, groupby
from histbook import bin as hbin
import numpy as np
import pandas as pd

worker_class = """
class Worker(object):

    Columns = ["nJet","nMuon","nElectron","Jet.pt", "Muon.pt"]

    def run(self, events, job):
        job.fill(nJet=events.nJet)
	job.fill(nElectron=events.nElectron)
	job.fill(nMuon=events.nMuon)
        job.fill(JetPt = events.Jet.pt)
	job.fill(MuonPt = events.Muon.pt)
"""

session = Session()

njets = Hist(hbin("nJet", 20, 0, 20))
nmuon = Hist(hbin("nMuon", 20, 0, 20))
nelectron = Hist(hbin("nElectron", 20, 0, 20))
muon_pt = Hist(hbin("MuonPt", 70, 0., 700.))
jet_pt = Hist(hbin("JetPt", 70, 0., 700.))


dataset = "QCD_HT200to300_PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1_NANOAODSIM"
#dataset = "JetHT_Run2016H_05Feb2018_ver2-v1_NANOAOD"

class Callback:
	def on_exception(self, wid, info):
		print "Exception:", info
        

job = session.createJob(dataset,
			user_callback = Callback(),
		    worker_class_source = worker_class, 
		    histograms = [njets, nmuon, nelectron, muon_pt, jet_pt]
)
job.run()
runtime = job.TFinish - job.TStart
nevents = job.EventsProcessed
print "%s: %.6fM events, %.6fM events/second" % (dataset, float(nevents)/1000000, nevents/runtime/1000000)

print muon_pt.pandas()
