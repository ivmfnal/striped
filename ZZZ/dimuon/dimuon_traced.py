from Fcn import cosh
import math
from trace import Tracer
import sys

def columns():
    return ["Muon.pt", "Muon.eta", "Muon.phi", "Muon.q"]

def init(dataset, my_id):
    dataset.Muon.p = dataset.Muon.pt * cosh(dataset.Muon.eta)

def run(events):

    T = Tracer()

    for e in events:
        with T["event"]:
            if len(e.Muon) > 1:
                with T["event/dimuon"]:
                    with T["event/dimuon/make_pairs"]:
                        pairs = list(e.Muon.pairs())
                    for m1, m2 in pairs:
                        with T["event/dimuon/pair"]:
                            if m1.q * m2.q < 0 and m1.p < 1000 and m2.p < 1000:     # Cuts
                                with T["event/dimuon/pair/opposite"]:
                                    M12 = math.sqrt(2*m1.pt*m2.pt*(math.cosh(m1.eta-m2.eta) - math.cos(m1.phi-m2.phi)))  
                                    if M12 < 140 and M12 > 60:
                                        with T["event/dimuon/pair/opposite/mass_in_range"]:
                                            yield (M12, m1.p, m2.p) 
    sys.stderr.write(T.formatStats()+"\n")
