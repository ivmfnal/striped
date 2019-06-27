import numpy as np

def bitmask(mask, length):
    if not mask.dtype is np.bool:
        bm = np.full((length,), False)
        bm[mask] = True
        mask = bm
    return mask

class Filterable(object):

    def apply_filter(self, filter):
        if isinstance(filter, QABranchFilter):
            return self.apply_row_filter(filter)
        elif isinstance(filter, QAEventFilter):
            return self.apply_event_filter(filter)

    def apply_event_filter(self, filter):
        raise NotImplementedError("apply_event_filter not implemeted for %s" % (self,))
        
    def apply_row_filter(self, filter):
        raise NotImplementedError("apply_row_filter not implemeted for %s" % (self,))
        
    def cut(self, mask):
        return self.filter(mask)(self)

class QAFilter(object):

    def __init__(self, parent, mask, size_array):
        self.Parent = parent
        self.SizeArray = size_array
        self.Length = sum(size_array)
        self.Nin = np.count_nonzero(mask) if mask.dtype is np.bool else len(mask)
        assert len(mask.shape) == 1
        assert len(size_array.shape) == 1
        self.Mask = bitmask(mask, self.Length)
        
    def __call__(self, subject):    
        if isinstance(subject, np.ndarray):
            return subject[self.Mask]
        else:
            return self.apply(subject)

    def __and__(self, another):
        return self.combine(another, lambda x, y: x * y)
        
    def __mul__(self, another):
        return self.combine(another, lambda x, y: x * y)
        
    def __or__(self, another):
        assert another.__class__ is self.__class__
        return self.combine(another, lambda x, y: x + y)
        
    def __add__(self, another):
        assert another.__class__ is self.__class__
        return self.combine(another, lambda x, y: x + y)
        
    @property
    def mask(self):
        return self.Mask
        
    @property
    def counts(self):
        return self.Nin, np.sum(self.Mask)
        
    @property
    def ratio(self):
        nin, n = self.counts
        if nin == 0:
            nin, n = 1, 1
        return float(n)/float(nin)
        
class QAEventFilter(QAFilter):

    def __init__(self, parent, mask):
        size_array = np.ones_like(mask)
        QAFilter.__init__(self, parent, mask, size_array)
        
    def expandMask(self, new_size_array):       # create a mask for a branch array
        assert len(new_size_array) == len(self.Mask)
        nn = np.sum(new_size_array)
        mask = np.empty((nn,), dtype=np.bool)
        j = 0
        for x, n in zip(self.Mask, new_size_array):
            mask[j:j+n] = x
            j += n
        return mask
        
    def combine(self, another, op = lambda x, y: x * y):
        #print "combine: self.Event:", self.Event,"  another.Event:", another.Event
        assert len(self.SizeArray) == len(another.SizeArray)
        if isinstance(another, QAEventFilter):
            assert np.all(self.SizeArray == another.SizeArray)
            return QAEventFilter(self.Parent, op(self.Mask, another.Mask))
        else:
            mask = self.expandMask(another.SizeArray)
            return QAEventFilter(self.Parent, op(mask, another.Mask))
            
    def apply(self, subject):
        if hasattr(subject, "apply_event_mask"):
            #print subject, "has attr apply_event_mask"
            return subject.apply_event_mask(self.Mask)
        else:
            return subject.apply_event_filter(self)

class QABranchFilter(QAFilter):

    def __init__(self, parent, mask, size_array):
        QAFilter.__init__(self, parent, mask, size_array)
        
    def reduceMask(self, operation="any"):       # create a mask for an event array
        operation = np.any if operation == "any" else np.all
        nn = len(self.SizeArray)
        mask = np.empty((nn,), dtype=np.bool)
        j = 0
        for i, n in enumerate(self.SizeArray):
            mask[i] = False if n <= 0 else operation(self.Mask[j:j+n])
            j += n
        return mask

    def combine(self, another, op = lambda x, y: x * y):
        #print "combine: self.Event:", self.Event,"  another.Event:", another.Event
        assert len(self.SizeArray) == len(another.SizeArray)
        if isinstance(another, QABranchFilter):
            assert self.Parent is another.Parent, "Can not compine filters for different branches"
            return QABranchFilter(self.Parent, mask, op(self.Mask, another.Mask))
        else:
            mask = another.expandMask(self.SizeArray)
            return QAEventFilter(self.Parent, op(self.Mask, mask))

    def apply(self, subject):
        return subject.apply_row_filter(self)
