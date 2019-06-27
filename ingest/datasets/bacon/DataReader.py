import uproot, numpy as np, re
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

class FlattenedJaggedJaggedUprootAray(UprootArrayBase):
    def __init__(self, dtype, shape, jagged_jagged):
        UprootArrayBase.__init__(self, dtype, shape)
        empty = np.array([], dtype=dtype)
        self.Data = [np.asarray(np.concatenate(lst), dtype=dtype) if len(lst)>0 else empty for lst in jagged_jagged]
        self.Sizes = np.array(map(len, self.Data), dtype=np.uint32)
        
    def stripesAndSizes(self, gsizes):
        i = 0
        max_stripe_size = 0
        for gs in gsizes:
            sizes = self.Sizes[i:i+gs]
            stripe = np.concatenate(self.Data[i:i+gs])
            stripe = np.asarray(stripe, dtype=self.DType)
            max_stripe_size = max(max_stripe_size, len(stripe.data))
            yield stripe, sizes
            i += gs

    def hasSizes(self):
        return True
        
class Counter(UprootArrayBase):

    def __init__(self, array, dtype):
        UprootArrayBase.__init__(self, dtype, (None,))
        sizes = []
        n_per_event = []
        for list_for_event in array:
            sizes_for_event = map(len, list_for_event)
            sizes.append(np.array(sizes_for_event, dtype=dtype))
            n_per_event.append(len(sizes_for_event))
	#print "n_per_event: %s" % (n_per_event,)
        self.Sizes = np.array(n_per_event, dtype=np.int64)
        self.Data = np.concatenate(sizes)
        
    def stripesAndSizes(self, gsizes):
        i = 0
        max_stripe_size = 0
        for gs in gsizes:
            sizes = self.Sizes[i:i+gs]
            stripe = self.Data[i:i+gs]
            stripe = np.asarray(stripe, dtype=self.DType)
            max_stripe_size = max(max_stripe_size, len(stripe.data))
            yield stripe, sizes
            i += gs
        
    def hasSizes(self):
        return True
        

class DataReader(BaseDataReader):

    DoubleJaggeds = [
	"AK4Puppi.pfCands", "AK8Puppi.pfCands", "AddAK8Puppi.svtx", "AddCA15Puppi.svtx", "CA15Puppi.pfCands", "AK4CHS.pfCands"
	]
    

    def __init__(self, file_path, schema):
        self.Path = file_path
        self.Tree = uproot.open(file_path)["Events"]
        self.BranchSizeArrays = {}
        self.Schema = schema
        self.T = Tracer()
        self.Converted = {}
        
    def profile(self):
	return None
                    
    def reopen(self):
        self.Tree = uproot.open(self.Path)["Events"]
        pass
        
    def nevents(self):
        return self.Tree.numentries
        
    def branchSizeArray(self, bname):
        bdesc = self.Schema["branches"][bname]
        counts = self.BranchSizeArrays.get(bname)
        if counts is None:
		try:
			a = Tree[bname].array()
			counts = np.asarray(a, dtype=np.uint32)
		except:
			counts = None
		self.BranchSizeArrays[bname] = counts
			
        return counts

    def stripesAndSizes(self, groups, bname, attr_name, attr_desc):
        src = attr_desc["source"]
        if src.startswith(":count_of:"):
            #print "converting %s..." % (src,)
            k = src[10:]
            with self.T["stripesAndSizes/counter_field"]:
                arr = Counter(self.Tree[k].array(), attr_desc["dtype"])
        else:
            b = self.Tree[src]
            if "%s.%s" % (bname, attr_name) in self.DoubleJaggeds:
                with self.T["stripesAndSizes/special_field"]:
                    array = b.array()
                    arr = FlattenedJaggedJaggedUprootAray(attr_desc["dtype"], (None,), array)
            else:
                with self.T["stripesAndSizes/regular_field"]:
                    arr = UprootArray.create(b)
        if bname is None:
            # attr
            with self.T["attr/stripes"]:
                parts = list(arr.stripes(groups))
            for part in parts:
                    yield part, None
        else:
            # branch attr
            #size_array = self.branchSizeArray(bname)
            #print "%s.%s..." % (bname, attr_name)
            with self.T["branch_atts/stripes"]:
                parts = list(arr.stripesAndSizes(groups))
            for data, size in parts:
                yield data, None        # we do not have any branch attributes with depth > 1
            
            
            
            
        
        
        
