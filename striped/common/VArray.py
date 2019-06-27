import numpy as np

_IndexDType = np.uint32

def generate_jagged_rec(depth, maxdata, maxbranch, xstart=0):
    n = random.randint(0,maxbranch)
    if depth == 0:
        segment = np.arange(n) + xstart
        return xstart + n, segment
    else:
        out = []
        for _ in xrange(n):
            xstart, part = generate_jagged_rec(depth-1, maxdata, maxbranch, xstart=xstart)
            out.append(part)
            if maxdata is not None and xstart > maxdata:    break
        return xstart, out

def generate_jagged(n, depth, maxdata, maxbranch):
    out = []
    x = 0
    while (n is None or n > 0) and (maxdata is None or x < maxdata):
        x, item = generate_jagged_rec(depth-1, maxdata, maxbranch, x)
        out.append(item)
    return x, out
    


def _analyze_rec(depth, data):

    if depth == 0:
        return [], [data] if len(data) else []

    elif depth == 1:
        sizes = map(len, data)
        return [sizes], [a for a in data if len(a)]

    else:
        top_sizes = []
        down_sizes = [[] for _ in xrange(depth-1)]
        data_segments = []
        for arr in data:
            top_sizes.append(len(arr))
            down, data = _analyze_rec(depth-1, arr)
            data_segments += data
            if down_sizes is None:
                down_sizes = down
            else:
                for x, d in zip(down_sizes, down):
                    x += d
        return [top_sizes] + down_sizes, data_segments

def _analyze(depth, dtype, data):
    """
     convert list of lists of lists like
     [
        [
           [a,b]
           [c]
        ]
        []
        [
           [d]
        ]
     ]

     to tuple:
     (
         [                  # skeleton index
             [2,0,1]
             [2,1,1]
         ],
         [a,b,c,d]          # flat data array
    )

    """
    sizes, data_segments = _analyze_rec(depth, data)
    sizes = [
        np.array(level, dtype=_IndexDType)
        for level in sizes
    ]
    if len(data_segments):
        flat = np.concatenate(data_segments)
    else:
        flat = np.array([], dtype=dtype)
    return sizes, flat

def _expand_skeleton_index(skeleton_index):
    if not skeleton_index:
        return []
    layer_0 = []
    i = 0
    sizes = skeleton_index
    sizes = sizes[::-1]
    for n in sizes[0]:
        layer_0.append((n,0,0,i,i+n))
        i += n
    index = [np.array(layer_0, dtype=_IndexDType)]
    last_index_layer = layer_0
    for size_layer in sizes[1:]:
        iinx = 0
        idata = 0
        new_index_layer = []
        for n in size_layer:
            ii0 = iinx
            ii1 = iinx + n
            id0 = idata
            id1 = idata
            if n > 0:
                id1 = last_index_layer[ii0+n-1][-1]
            new_index_layer.append((n, ii0, ii1, id0, id1))
            iinx = ii1
            idata = id1
        index.insert(0, np.array(new_index_layer))
        last_index_layer = new_index_layer
    return index

def _expand_flat_index_rec(depth, flat_index, i0, ntop=None, imax=None):
    if depth == 1:
        n = imax-i if imax is not None else ntop
        return [list(flat_index[i0:i0+n])], i0+n
    top = []
    down_inx = [[] for _ in xrange(depth-1)]        # can not use [[]] * (depth-1) because that will be a list of multiple instances of the same empty list !
    i = i0
    while (imax is None or i < imax) and (ntop is None or ntop > 0):
        ndown = flat_index[i]
        down_add, i = _expand_flat_index_rec(depth-1, flat_index, i+1, ntop=ndown)
        #print "_expand_flat_index_rec: adding down index"
        for d, a in zip(down_inx, down_add):
            if len(a):
                #print type(d), d, type(a), a
                d += a
        top.append(ndown)
        if ntop is not None:    ntop -= 1
    return [top] + down_inx, i
            
def _expand_flat_index(depth, flat_index):
    assert depth > 0 or flat_index is None, "Index for array with depth=0 must be None"
    if depth == 0:
        return None
    flat_index = np.array(flat_index, dtype=_IndexDType)
    if depth == 1:
        return [flat_index]
    index, _ = _expand_flat_index_rec(depth, flat_index, 0, imax=len(flat_index))
    return _expand_skeleton_index(index)
        
def _compact_index_data_rec(depth, index, data, dtype):
    if depth == 0:
        return None, data
    elif depth == 1:
        inx = index[0]
        if inx[0,3] == 0 and inx[-1,4] == len(data) and inx[:-1,4] == inx[1:,3]:
            return index, data      # already compacted
        segments = []
        out_index = []
        out_id0 = 0
        for n,_,_,id0,id1 in inx:
            out_id1 = out_id0 + n
            out_index.append((n,0,0,out_id0,out_id1))
            if id0 > id1:   segments.append(data[id0:id1])
            out_id0 = out_id1
        out_data = np.concatenate(segments) if len(segments) else np.array([], dtype)
        out_index = np.array(out_index, dtype = _IndexDType)
        return out_index, out_data
    # compact recursively
    top = index[0]
    compact_down, compact_data = _compact_index_data_rec(depth - 1, index[1:], data, dtype)
    # build compact top index layer
    ix0 = 0
    next_level = compact_down[0]
    top_out = []
    for n,_,_,_,_ in top:
        ix1 = ix0 + n
        id0, id1 = next_level[ix0,3],next_level[ix1,4]
        top_out.append((n, ix0, ix1, id0, id1))
    return [np.array(top_out, dtype=IndexDType)] + compact_down, compact_data
        
class VArray(object):
    
    IndexDType = np.uint64
    
    def __init__(self, depth, dtype, flat=None, index=[]):
        
        assert (flat is None) == (index is None), "Both index and flat array must be specified, or both unspecified"
        assert (index is None) or (len(index) == depth), "Index length (%d) is not consistent with the erray depth (%d)" % (len(index), depth)

        self.Depth = depth
        self.DType = dtype        
        self.FlatArray = flat
        self.Index = index           # expanded index, int32 ndarray
        
    @property
    def flat(self):
        # returns whole data array, ignoring filtering. 
        # This is still good for iteration, but not for striping
        return self.FlatArray       

    def _segments_gen(self):
        if self.Depth == 0:
            yield self.FlatArray
        elif self.Depth == 1:
            for _,_,_,id0,id1 in self.Index[0]:
                yield self.FlatArray[id0:id1]
        else:
            for v in self:
                for s in v.__segments_gen():
                    yield s

    def isCompact(self):
        if self.Depth == 0:
            return True
        else:
            top = self.Index[0]
            id0, id1 = top[0,3], top[-1,4]
            if id0 != 0 or id1 != len(self.FlatArray):   return False        # margins in data array
            for layer in self.Index:
                #print "layer:", layer
                if np.any(layer[1:,3] != layer[:-1,4]):     return False        # gap in data 
                if np.any(layer[1:,1] != layer[:-1,2]):     return False        # gap in index
        return True
            
    def compact(self, in_place=False):
        if not self.isCompact():
            compact_index, compact_data = _compact_index_data_rec(self.Depth, self.Index, self.FlatArray, self.DType)
            if in_place:
                self.Index = compact_index
                self.FlatArray = compact_data
            else:
                return VArray(self.Depth, self.DType, flat=compact_data, index=compact_index)
        else:
            return self
        
    @staticmethod
    def fromJagged(depth, dtype, data):
        # assume the input array has same depth
        
        skeleton_index, flat = _analyze(depth, dtype, data)
        index = _expand_skeleton_index(skeleton_index)
        
        return VArray(depth, dtype, flat=flat, index=index)
        
    @staticmethod
    def fromArray(data, dtype=None):
        dtype = dtype or data.dtype
        v = VArray(0, dtype)
        v.FlatArray = np.asarray(data, dtype=dtype)
        return v
        
    @staticmethod
    def fromVArrayAndCounts(varray, counts):
        # counts is 1-dim integer array or list
        assert isinstance(counts, (list, tuple)) or \
                isinstance(counts, np.ndarray) and len(counts.shape) == 1
        if varray.Depth == 0:
            new_top = []
            id0 = 0
            for n in counts:
                new_top.append((n, 0, 0, id0, id0+n))
                id0 += n
        else:
            old_top = varray.Index[0]
            ix0 = 0
            id0 = old_top[ix0][3]
            new_top = []
            for n in counts:
                ix1 = ix0 + n
                id0 = old_top[ix0][3]
                id1 = old_top[ix1-1][4]
                new_top.append((n, ix0, ix1, id0, id1))
                ix0 = ix1
        return VArray(varray.Depth+1, varray.DType, flat=varray.flat, index=[new_top] + varray.Index)
        
    @staticmethod
    def fromFlat(depth, flat_index, data):
        index = _expand_flat_index(depth, flat_index)
        return VArray(depth, data.dtype, index=index, flat=data)
        
    def filter(self, mask):
        if self.Depth == 0:
            return self.FlatArray[mask]
        else:
            top_index = self.Index[0][mask]
            return VArray(self.Depth, self.DType, 
                flat = self.FlatArray, 
                index = [top_index] + self.Index[1:])
    
    def __len__(self):
        if self.Depth == 0:
            return len(self.flat)
        else:
            return len(self.Index[0])
            
    def __getitem__(self, inx):
        if isinstance(inx, int):
            if self.Depth == 0:
                return self.FlatArray[inx]
            elif self.Depth == 1:
                _,_,_,id0,id1 = self.Index[0][inx]
                return self.FlatArray[id0:id1]
            else:
                layer = self.Index[0]
                _,ix0,ix1,_,_ = layer[inx]
                return VArray(self.Depth-1, self.DType, self.FlatArray, [self.Index[1][ix0:ix1]] + self.Index[2:])
        elif isinstance(inx, slice):
            if self.Depth == 0:
                return self.FlatArray[inx]
            else:
                inxlst = range(*inx.indices(len(self)))
                return self.filter(inxlst)
                
    def iterate(self):
        n = len(self)
        for i in xrange(n):
            yield self[i]
            
    def __iterate____(self):
        if self.Depth == 0:
            for x in self.FlatArray: yield x
        elif self.Depth == 1:
            for _, _, _, id0, id1 in self.Index[0]:
                yield self.FlatArray[id0:id1]
        else:
            for n, ix0, ix1, id0, id1 in self.Index[0]:
                inx1 = self.Index[1]
                vindex = [inx1[ix0:ix1]] + self.Index[2:]
                va = VArray(self.Depth-1, self.DType, flat = self.FlatArray, index = vindex)
                yield va
        
    def __iter__(self):
        return self.iterate()
        
    def asList(self):
        return list(self)
        
    @property
    def flatIndex(self):
        inx = self._flat_index()
        return None if inx is None else np.array(inx, dtype=_IndexDType)
        
    def dataSegment(self, make_compact = True):
        # returns portion of the FlatArray within the array view, which means it is possible that
        # it will include filtered out data. Use with caution!
        if self.Depth == 0:
            return self.FlatArray
        else:
            if make_compact:
                v = self.compact()
            else:
                v = self
            index = v.Index[0]
            return v.FlatArray[index[0,3]:index[-1,4]]
        
    def _flat_index(self):
        if self.Depth == 0:
            return None
        elif self.Depth == 1:
            out = [n for n,_,_,_,_ in self.Index[0]]
        else:
            out = []
            for v in self:
                out += [len(v)] + v._flat_index()
        return out
                
    def stripe(self, sizes):
        N = len(self)
        if isinstance(sizes, int):
            G = sizes
            lst = [G] * (N//G)
            if N % G:
                lst.append(N%G)
            sizes = lst
            
        v = self.compact()
        
        if v.Depth == 0:
            i = 0
            flat = v.FlatArray
            for g in sizes:
                yield flat[i:i+g], None
                i += g
        elif v.Depth == 1:
            i = 0
            flat = v.FlatArray
            index = v.Index[0]
            for g in sizes:
                i1 = i + g - 1
                id0 = index[i,3]
                id1 = index[i1,4]
                yield flat[id0:id1], index[i:i+g,0]
                i += g
        else:
            i = 0
            for g in sizes:
                segment = v[i:i+g]
                yield segment.dataSegment(make_compact=False), segment.flatIndex
                i += g
                
if __name__ == '__main__':
    import random, numpy as np, pprint
    random.seed(1001)


    import time, random, json
    from Stopwatch import Stopwatch

    class MyJSONEncoder(json.JSONEncoder):
        
        #def __init__(self, *params, **args):
        #    json.JSONEncoder(self, *params, **args)
            
        def default(self, obj):
            if isinstance(obj, (np.ndarray, VArray)):
                return list(obj)
            
    
    encoder = MyJSONEncoder(indent=4, sort_keys=True)
    
    def compare_jagged(a1, a2):
        return encoder.encode(a1) == encoder.encode(a2)
        
    def iterate_test():
        maxdata = 15
        with Stopwatch("creating", inline=True):
            ndata, arr = generate_jagged(10, 3, maxdata, 5)
        print "Total data length=", ndata
        
        #print "Data:"
        #pprint.pprint(arr)
        
        with Stopwatch("folding", inline=True):
            v = VArray.fromJagged(3, np.float, arr)
        #print "Index:", v.SIndex
        flat = v.flat
        print "Flat array:", v.flat
        print "Index:          "
        pprint.pprint(v.Index)
        
        print v.flatIndex
        
        print "original array:"
        print encoder.encode(arr)
        
        print "v array (%s)" % (v,)
        print encoder.encode(v)
        
        print compare_jagged(arr, v)
        
    def mask_test():
        ndata, arr = generate_jagged(3, 2, 60, 5)
        print len(arr), arr
        v = VArray.fromJagged(2, np.float, arr)
        n = len(v)
        mask = np.asarray(np.arange(n) % 2, np.bool)
        print mask
        
        print encoder.encode(v)
        
        v1 = v.filter(mask)
        print encoder.encode(v1)
        
    def scale_test():
        depth = 2
        with Stopwatch("generate", inline=False):
            ndata, arr = generate_jagged(100000, depth, 1000000, 10)
            print "max data:", ndata
            
        #pprint.pprint(arr)
            
        with Stopwatch("analyze", inline=True):
            v = VArray.fromJagged(depth, np.float, arr)
            
        with Stopwatch("flat index", inline=True):
            inx = v.flatIndex
            
        mask = np.asarray(np.arange(len(v)) % 2, np.bool)
        with Stopwatch("mask", inline=True):
            v1 = v.filter(mask)
            
    def stripe_test():
        depth = 3
        with Stopwatch("generate", inline=True):
            ndata, arr = generate_jagged(100000, depth, 10000000, 10)
        print "ndata=", ndata, "   items:", len(arr)
        
        #print encoder.encode(arr)
            
        with Stopwatch("analyze", inline=True):
            v = VArray.fromJagged(depth, np.float, arr)
        
        striped = []
        
        with Stopwatch("to stripes", inline=True):
            for d, s in v.stripe(10000):
                striped.append((d, s))

        with Stopwatch("assemble", inline=True):
            for d, s in striped:
                v = VArray.fromFlat(depth, s, d)
            
            
    stripe_test()
    
