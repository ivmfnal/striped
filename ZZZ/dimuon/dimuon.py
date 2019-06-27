from Q import cosh, sqrt, cos
import math

def columns():
    return ["Muon.pt", "Muon.eta", "Muon.phi", "Muon.q"]

def init(dataset, my_id):

    m = dataset.event.Muon
    m.p = m.pt * cosh(m.eta)
    
    pair = dataset.event.Muon.pair
    m1, m2 = pair
    pair.M = sqrt(2*m1.pt*m2.pt*(cosh(m1.eta-m2.eta) - cos(m1.phi-m2.phi)))  
    pair.C = m1.q * m2.q

def run(events, emit):
    for e in events:
        for pair in e.Muon.pairs.iterate():
            m1, m2 = pair.asTuple()
            if pair.C < 0 and m1.p < 1000 and m2.p < 1000:     
                M = pair.M  
                if M < 120 and M > 60:
                    emit("mass", M)
                    if M < 92 and M > 88:
                        emit("momentum", m2.p)
                    
                
