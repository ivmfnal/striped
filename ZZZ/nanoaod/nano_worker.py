import numpy as np

class Worker(object):

    Columns = ["Jet.pt", "Jet.eta", "Jet.jetId"]

    def __init__(self, params, bulk, job, database):
        self.Job = job
        self.Params = params
        self.Bulk = bulk

        self.Sum = 0.0
        self.N = 0
        
        self.First = True
        
    def frame(self, events):
    
        if self.First:
            self.Job.message("calibrations shape: %s" % (self.Bulk["calibrations"].shape,))
            self.First = False
    
        jets = events.Jet
        jets_pt = jets.pt

        filter = jets.filter((jets.eta < 2.5) * (jets.jetId > 0))
        filtered_pts = filter(jets_pt)
        
        sum_pt = sum(filtered_pts)
        n_filtered = len(filtered_pts)
        
        self.Sum += sum_pt
        self.N += n_filtered
        
    def end(self):
        return {
            "sum_pt":       self.Sum,
            "n_jets":       self.N
        }
        
class Accumulator:
    
    def __init__(self, params, bulk, job_interface, db_interface):
        self.Data = None
                
    def add(self, data):
        if self.Data is None:
            self.Data = data.copy()
        else:
            for k, v in data.items():
                self.Data[k] += v
        
    def values(self):
        return self.Data
