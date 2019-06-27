from Fcn import cosh
import math
from trace import Tracer

out_file = None
dataset_name = None

def columns():
    return ["Muon.pt", "Muon.eta", "Muon.phi", "Muon.q"]

def init(dataset, my_id):
    global out_file, dataset_name
    dataset_name = dataset.DatasetName
    out_file = "/tmp/%s_%d.csv" % (dataset_name, my_id)
    dataset.Muon.p = dataset.Muon.pt * cosh(dataset.Muon.eta)

def run(events):
    with open(out_file, "w") as outf:
        for e in events:
            outf.write("%s,%d,NMuons,%f\n" % (dataset_name, e._EventID,len(e.Muon)))
            dimuon_pair = False
            for m1, m2 in e.Muon.pairs():
                    if m1.q * m2.q < 0 and m1.p < 1000 and m2.p < 1000:
                        dimuon_pair = True
                        break
            if dimuon_pair:
                outf.write("%s,%d,DimuonPair,1.0\n" % (dataset_name, e._EventID))
    
                        
