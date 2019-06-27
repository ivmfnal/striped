import numpy as np
from .H import Hist

class HAggregator(object):  

    # this object wraps the histbook histogram at the receiving end (the job)

    def __init__(self, histogram, inputs=None, display=None, constants=None):
        self.H = histogram
        #self.Display = display		not used
        #self.Constants = constants   not used
            
    def inputs(self):
        return self.H.fields
        
    def descriptor(self):
	return self.H.meta()

    def add(self, dump):
	counts = Hist.deserialize_counts(dump)
	self.H.add_counts(counts)
        
    @property
    def histogram(self):
        return self.H
        

class HCollector(object):

    # this object wraps the histbook histogram at the sending end (the job)

    def __init__(self, desc):
        # hist_descriptor is what is produced by HBHistogramCompiler.descriptor()
        self.H = Hist.empty_from_meta(desc)
        self.Inputs = set(self.H.fields)
        self.T = None

    def isInput(self, inp):
        return inp in self.Inputs

    def inputs(self):
        return self.Inputs
        
    def fill(self, data_dict):
	self.H.fill(data_dict)
    
    def dump(self, clear=True):
	dump = self.H.serialize_counts()
        if clear:
            self.H.clear()
        return dump
        
    @property
    def histogram(self):
        return self.H
   
    
    