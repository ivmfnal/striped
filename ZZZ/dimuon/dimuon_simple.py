from Fcn import cosh
import math
from trace import Tracer

def columns():
    return ["Muon.pt", "Muon.eta", "Muon.phi", "Muon.q"]

def init(dataset, my_id):
    pass

def run(events):
    for e in events:
        for m1, m2 in e.Muon.pairs():
            if True or m1.q * m2.q < 0:     # Cuts
                M12 = math.sqrt(2*m1.pt*m2.pt*(math.cosh(m1.eta-m2.eta) - math.cos(m1.phi-m2.phi)))  
                if M12 < 120 and M12 > 60:
                    yield (M12,)
