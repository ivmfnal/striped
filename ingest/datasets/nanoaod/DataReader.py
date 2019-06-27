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

class DataReader(BaseDataReader):

    def __init__(self, file_path, schema):
        self.Path = file_path
	#print "DataReader.__init__(%s)" % (file_path,)
        self.Tree = uproot.open(file_path)["Events"]
        self.BranchSizeArrays = {}
        self.Schema = schema
        self.T = Tracer()
        self.Converted = {}
        
    def profile(self):
        file_keys = self.Tree.keys()
	schema_keys = [desc["source"] for name, desc in self.Schema["attributes"].items()
				if desc.get("optional") == "yes"]
	metadata = { k: (k in file_keys) for k in schema_keys }
	return metadata
	
                    
    def reopen(self):
        self.Tree = uproot.open(self.Path)["Events"]
        pass
        
    def nevents(self):
        return self.Tree.numentries
        
    def branchSizeArray(self, bname):
	#print "branchSizeArray(%s, %s, %s)" % (bname, attr_name, attr_desc)
        bdesc = self.Schema["branches"][bname]
        counts = self.BranchSizeArrays.get(bname)
        if counts is None:
            fn, fdesc = bdesc.items()[0]
            try:
                with self.T["branchSizeArray/array_counts"]:
                    arr = self.Tree[fdesc["source"]].array()
                    counts = np.array(map(len, arr), dtype=np.uint32)
            except:
                print "Error getting size array for branch %s using field %s. Array:%s" % (bname, fn, arr)
                raise
            #self.BranchSizeArrays[bname] = counts
        return counts

    def stripesAndSizes(self, groups, bname, attr_name, attr_desc):
	#print "stripesAndSizes(%s, %s, %s)" % (bname, attr_name, attr_desc)
        src = attr_desc["source"]
        try:	b = self.Tree[src]
	except:
		optional = attr_desc.get("optional","no") == "yes"
		if not optional:
			raise
		n = max(groups)
		zeros = np.zeros((n,), dtype=attr_desc["dtype"])
		for g in groups:
			#print "stripesAndSizes yield 1"
			yield zeros[:g], None
	else:
		#print "UprootArray.create..."
		arr = UprootArray.create(b)
		#print "UprootArray.create done"
		if bname is None:
		    # attr
		    with self.T["attr/stripes"]:
			#print "arr.stripes..."
			parts = list(arr.stripes(groups))
			#print "arr.stripes done"
		    for part in parts:
			    #print "stripesAndSizes yield 2"
			    yield part, None
		else:
		    # branch attr
		    #size_array = self.branchSizeArray(bname)
		    #print "%s.%s..." % (bname, attr_name)
		    with self.T["branch_atts/stripes"]:
			parts = list(arr.stripesAndSizes(groups))
		    for data, size in parts:
			#print "stripesAndSizes yield 3"
			yield data, None        # we do not have any branch attributes with depth > 1
            
            
            
            
        
        
        
