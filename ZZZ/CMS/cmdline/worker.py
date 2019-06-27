import math, time
from Q import sqrt

class Worker(object):

    def columns(self):
        return ["Muon.p4"]

    def init(self, dataset, my_id):
        event = dataset.event
        event.nmuons = event.Muon.length
        dataset.emit("nmu", event.nmuons)
        
        mupair = event.Muon.pair
        m1, m2 = mupair
        sump = m1.p4 + m2.p4
        mupair.M = sqrt(sump[3]*sump[3] - sump[0]*sump[0] - sump[1]*sump[1] - sump[2]*sump[2])
        
        dataset.emit("mass", mupair.M)
        dataset.emit("e1", m1.p4[3])
        dataset.emit("e2", m2.p4[3])

        dataset.filter(event.Muon.length>1)

        
    def run(self, events, emit):
        #return
        for e in events:
            #pass
            #time.sleep(10)
            #emit("nmuons", e.nmuons)
            for mupair in e.Muon.pairs:
                emit("mu_mu_mass", mupair.M)
            #emit("nmu", e.nmuons)
            pass
        
