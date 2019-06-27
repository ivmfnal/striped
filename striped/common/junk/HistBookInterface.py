#import cPickle, base64, 
import json
import numpy as np



class HBHistogramAggregator(object):  

    # this object wraps the histbook histogram at the receiving end (the job)

    def __init__(self, histogram, inputs, display, constants):
        self.H = histogram
        self.Display = display
        self.Constants = constants
        if isinstance(inputs, (list, tuple)):
            self.Mapping = {x:x for x in inputs}
        elif isinstance(inputs, dict):
            self.Mapping = inputs
        else:
            raise ValueError("Histogram inputs must be either list of strings or a dictionary")
            
    def inputs(self):
        return self.Mapping.keys()
        
    def descriptor(self):
        dct = {
            "historgam":        self.H.cleared().tojson(),
            "mapping":          self.Mapping,
            "constants":        self.Constants
        }
        return dct

    def add(self, dump):
        from histbook import Hist
        # dump is what is returned by HBHistogramCollector.dump
        h = Hist.fromjson(json.loads(dump))
        #print "HBHistogramAggregator.add: delta content: %s" % (h._content,)
        self.H += h
        
    @property
    def histogram(self):
        return self.H
        

class HBHistogramCollector(object):

    # this object wraps the histbook histogram at the sending end (the job)

    def __init__(self, desc):
        from histbook import Hist
        # hist_descriptor is what is produced by HBHistogramCompiler.descriptor()
        self.H = Hist.fromjson(desc["historgam"])
        self.Mapping = desc["mapping"]              # maps historgam input name to stream name
        self.Constants = desc["constants"]
        self.Inputs = set(self.Mapping.keys())
        self.T = None

    def isInput(self, inp):
        return inp in self.Inputs

    def inputs(self):
        return self.Inputs
        
    def replicate_constant(self, shape, value):
        v = np.array(value)
        x = np.empty(shape, dtype=v.dtype)
        x.flat[0] = value
        x.strides = (0,)*len(shape)
        return x
        
        
    def fill(self, data_dict):
    
        scalars = {}
        arrays = {}
        array_length = None
        
        for n, v in data_dict.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                l = len(v)
                if array_length is not None and l != array_length:
                    raise ValueError("Inconsistent data length for histogram %s, item name %s, item length=%d, expected length=%s" % (self.H, n, l, array_length))
                array_length = l
                arrays[n] = v
            else:
                scalars[n] = v
                    
        if not arrays:
            # all scalars
            arrays = scalars
            array_length = 1
        
        elif array_length > 0:
            # replicate scalars
            for sn, sv in scalars.items():
                x = np.array(sv)
                x = np.empty((array_length,), dtype=x.dtype)
                x.flat[0] = sv
                x.strides = (0,)
                arrays[sn] = x

        if arrays and array_length > 0:
            #for n, a in arrays.items():
            #    print "len of %s = %d" % (n, len(a))
            #print "fill"
            try:
                if self.T is not None:
                    self.T.begin("hist/fill")
                self.H.fill(**arrays)
                if self.T is not None:
                    self.T.end("hist/fill")
            except:
                #print ("error in H.fill. arrays=%s, scalars=%s" % (arrays,scalars))
                raise
    def dump(self, clear=True):
        dump = json.dumps(self.H.tojson())
        if clear:
            self.H.clear()
        return dump
        
    @property
    def histogram(self):
        return self.H
   
    
    
