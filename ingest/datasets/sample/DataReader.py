import numpy as np, re, yaml
from striped.ingestion import UprootArray, UprootArrayBase, BaseDataReader
from striped.common import Tracer

def stripeAttr(groups, array):
    i = 0
    for n in groups:
        yield array[i:i+n]
        i += n

def stripeBranchAttr(groups, counts, array):
    i = 0
    j = 0
    for n in groups:
        m = sum(counts[i:i+n])
        yield array[j:j+m]
        j += m
        i += n


class DataReader(BaseDataReader):

    def __init__(self, file_path, schema):
        self.Schema = schema
        self.T = Tracer()
        self.Config = yaml.load(open(file_path, "r"))
        self.NEvents = self.Config["NEvents"]
        self.NBPerEvent = self.Config["NBPerEvent"]
        
    def profile(self):
        return None
                    
    def reopen(self):
        pass
        
    def nevents(self):
        return self.NEvents
        
    def branchSizeArray(self, bname):
        return np.repeat(self.NBPerEvent[bname], self.NEvents)
        
    def stripesAndSizes(self, groups, bname, attr_name, attr_desc):
        dtype = attr_desc["dtype"]
        n_per_event = 1 if bname is None else self.NBPerEvent[bname]
        for g in groups:
            arr = np.asarray(np.random.random((n_per_event*g,))*10.0, dtype=dtype)
            print bname, attr_name  #, arr
            yield arr, None

        
