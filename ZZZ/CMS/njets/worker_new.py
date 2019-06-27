import math, time
from Q import sqrt

class Worker(object):

    def columns(self):
        return ["Muon.p4"]

    def process(self, events, emit):
        #print events.Muon.p4[:,3]
        emit(mu_px = events.Muon.p4[:,0],
            mu_py = events.Muon.p4[:,1],
            mu_pz = events.Muon.p4[:,2],
            mu_e = events.Muon.p4[:,3])
        
