import math, time
from Q import sqrt

class Worker(object):

    def columns(self):
        return ["NJets"]

    def init(self, dataset, my_id):
        dataset.emit("njets", dataset.event.NJets)

        
    def run(self, events, emit):
        #return
        for e in events:
            pass
        
