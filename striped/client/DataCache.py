from striped.common import synchronized, Lockable
import numpy as np
from collections import OrderedDict

class DataCache(Lockable):

    def __init__(self, max_size, max_keys = None):      
        Lockable.__init__(self)
        self.MaxSize = max_size
        self.Size = 0
        self.MaxKeys = max_keys
        self.Cache = OrderedDict()     

    @synchronized
    def get(self, key, raise_exception=False):
        data = self.Cache.get(key, None)
        if data is None:
            if raise_exception:
                raise KeyError(key)
        else:
            # bump the key
            del self.Cache[key]
            self.Cache[key] = data
        return data
    
    @synchronized
    def cached(self, key):
        return key in self.Cache
        
    def _item_size(self, data):
        size = 1
        if isinstance(data, np.ndarray):
            size = len(data.data)
        else:
            try:    size = max(1, len(data))
            except: size = 1
        return size
            
    @synchronized
    def store(self, key, data):
        size = self._item_size(data)
        if self.Size + size > self.MaxSize or (self.MaxKeys is not None and len(self.Keys) >= self.MaxKeys):
            self.purge(self.MaxSize - size)
        self.Cache[key] = data
        self.Size += size
        
    @synchronized
    def purge(self, size_target):
        #print "DataCache: size=", self.Size
        while len(self.Cache) > 0 and (
                (self.MaxKeys is not None and len(self.Cache) >= self.MaxKeys) 
                or self.Size > size_target):
            key, data = self.Cache.popitem(last=False)
            #print "DataCache: evicting:", key
            size = self._item_size(data)
            self.Size = max(self.Size - size, 0)


