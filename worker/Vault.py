import numpy as np
from striped_c_tools import pairs as make_pairs
from QAFilters import Filterable

def make_segments(size_array):
    if len(size_array) == 0:
        return np.array([], dtype=np.int)
    s = np.empty((len(size_array), 2), dtype=np.int64)
    np.cumsum(size_array, out=s[:,1])       # segment ends
    s[1:,0] = s[:-1,1]          # segment starts, begins with 0
    s[0,0] = 0
    return s


class Stripe:
    def __init__(self, size_array, segments, data):
        self.SizeArray = size_array
        self.Data = data
        self.Segments = segments
        
    @property
    def values(self):
        return self.Data
        
    data = values       # synonym

    # all muon.p4s in muon pairs for event 123:
    # muon_pairs_vault["p4"][:][index(123):index(124)] - shape: (npairs, 2, 4)
    # all muon.px in muon pairs for event 123:
    # muon_pairs_vault["p4"][:,0][index(123):index(124)] - shape: (npairs, 2)
    # pz for second muon in pair
    # muon_pairs_vault["p4"][1][2]
    def __getitem__(self, inx):
        if isinstance(inx, (int, slice)):
            inx = (inx,)
        inx = (slice(None),) + inx
        out = self.Data[inx]
        return out
        
    @property
    def size(self):
        return self.SizeArray
        
    def row_mask(self, mask):     # applies the mask to the data and returns the ndarray
        return self.Data[mask]
        
    def apply_event_mask(self, mask):     # applies the event mask and returns new Stripe
        if self.SizeArray is None:
            return self.Data[mask]
        else:
            data_segments = [self.Data[start:stop] for i, (start, stop) in enumerate(self.Segments) if mask[i]]
            if len(data_segments):
                return np.concatenate(data_segments)
            else:
                return np.array([], dtype=self.Data.dtype).reshape((-1,)+self.Data.shape[1:])
                
    def segment(self, ievent):
        start, stop = self.Segments[ievent]
        return self.Data[start:stop]
        
        
class Vault(Filterable):

    # pairs vault: index_shape = (2,)
    # triplets: index_shape = (3,)

    def __init__(self, size_array=None, index_shape=(), stripes={}):
        self.SizeArray = size_array
        self.IndexShape = index_shape       # shape without the event dimenstion and without data shape
        self.Stripes = {}                   # attr name -> stripe. Stripe shape is: (along_events,) + index_shape + data_shape
        self.SizeArray = size_array
        self.Segments = None
        self.Length = None
        if size_array is not None:
            self.Segments = make_segments(size_array)
            self.Length = sum(size_array)
        for sn, s in stripes.items():
            self.addStripe(sn, s)
            
    def __str__(self):
        return "[Vault Length:%s SizeArray:(%s) IndexShape:%s]" % (self.Length, 
            None if self.SizeArray is None else len(self.SizeArray), self.IndexShape) 
    
    @property
    def primary(self):
        return len(self.IndexShape) == 0
            
    def addStripe(self, name, data_or_stripe):
        if isinstance(data_or_stripe, Stripe):
            data = data_or_stripe.values
        else:
            data = data_or_stripe
        if self.Length is None:
            self.Length = len(data)
        else:
            assert len(data) == self.Length, \
                "%s: incompatible stripe length in addStripe: %s, expected %s." % (
                    self, len(data), self.Length)
        assert data.shape[1:len(self.IndexShape)+1] == self.IndexShape
        #print "addStripe(%s): shape=%s" % (name, data.shape)
        self.Stripes[name] = data
        
    __setitem__ = addStripe
    
    def lengthForEvent(self, ievent):
        if self.SizeArray is None:  return 1
        else: return self.SizeArray[ievent]
        
    def stripe(self, name, *inx):
        # inx is optional indexes into IndexShape
        assert len(inx) <= len(self.IndexShape)
        data = self.Stripes[name]
        #print "Vault.stripe(%s, %s): data.shape:%s" % (name, inx, data.shape)
        if inx:
            s = (slice(None),)+inx                     # stripe[:,i1,i2]
            data = data[s]
            #print"     data.shape:", data.shape
        return Stripe(self.SizeArray, self.Segments, data)
        
    def __getitem__(self, name):
        return self.stripe(name)
        
    def event(self, attr_name, ievent):
        start, stop = self.Segments[ievent]
        return self.Stripes[attr_name][start:stop]
            
    def arrays(self, name, *inx):
        s = self.stripe(name, *inx).values
        if self.SizeArray is None:
            return list(s)
        else:
            return [s[start:end] for start, end in self.Segments]

    def expandArray(self, array):
        assert len(array) == len(self.SizeArray)
        new_length = sum(self.SizeArray)
        shape = (new_length,) + array.shape[1:]
        s = np.empty(shape, dtype=array.dtype)
        j = 0
        for n, x in zip(self.SizeArray, array):
                if n:
                        s[j:j+n] = x
                        j += n
        return s

    def makePairsVault(self):
        if not self.primary:
            raise NotImplementedError("Creating pairs from non-primary vaults is not implemented")

        pair_sizes = np.array([n*(n-1)/2 if n > 1 else 0 for n in self.SizeArray])
        nevents = self.Length
        L = sum(pair_sizes)

        data_vault = Vault(pair_sizes, (2,))

        for attr_name, stripe in self.Stripes.items():
            #print "makePairsVault: stripe.shape:", stripe.shape
            assert len(stripe.shape) <= 2            # can not do more than 1 data dimension yet
            if len(stripe.shape) == 1:
                # scalar
                pairs_stripe, _ = make_pairs(stripe.reshape((-1,)), self.SizeArray)
                pairs_stripe = pairs_stripe.T
            else:
                width = stripe.shape[-1]
                stripes = []
                for j in range(width):
                    d, _ = make_pairs(stripe[:,j], self.SizeArray)
                    #print "d shape:", d.shape
                    stripes.append(d)
                pairs_stripe = np.array(stripes)
                #print "makePairsVault: paits_stripe_shape:", pairs_stripe.shape
                #print "pairs stripe shape:", pairs_stripe.shape
                pairs_stripe = pairs_stripe.transpose((2,1,0))
            #print "makePairsVault: SizeArray:", self.SizeArray
            #print "makePairsVault: stripe shape:", pairs_stripe.shape
            data_vault.addStripe(attr_name, pairs_stripe)
        return data_vault
        
    def apply_event_mask(self, mask):
        #print "Vault.apply_event_mask: self:%s, mask:%d" % (self, len(mask))
        assert (self.SizeArray is None and (self.Length is None or len(mask) == self.Length)) \
             or (self.SizeArray is not None and len(mask) == len(self.SizeArray))
        if self.SizeArray is None:
            v = Vault()
        else:
            v = Vault(self.SizeArray[mask])
        for sn in self.Stripes.keys():
            v.addStripe(sn, self.stripe(sn).apply_event_mask(mask))
        return v    
        
    def apply_event_filter(self, filter):
        return self.apply_event_mask(filter.mask) 
        
    def apply_row_mask(self, mask):
        # compute new size array
        # there are 2 types of masks - boolean and index
        # in case we got the index one, convert it to boolean
        new_size_array = np.array([sum(mask[start:stop]) for start, stop in self.Segments])
        v = Vault(new_size_array, self.IndexShape)
        for sn, s in self.Stripes.items():
            v.addStripe(sn, s[mask])
        return v

    def apply_row_filter(self, filter):
        return self.apply_row_mask(filter.mask) 
