
import uproot, sys
import numpy as np

#print uproot.__version__




tree = uproot.open(sys.argv[1])["Events"]


def type_shape(t):
    def dimension(x):
        if isinstance(x, int):
            return x
        elif x is np.inf:
            return None
        else:
            raise ValueError("Can not decode dimension %s of type %s" % (x, type(x)))
    shape = []
    dtype = None
    if isinstance(t, np.dtype):
        return t, []     # scalar
    else:
        # assume the type object follows takes/to protocol
        takes, to = t.takes, t.to
        dtype, next_shape = type_shape(to)
        shape = [dimension(takes)] + next_shape
    
    #
    # make sure the shape variable dimensions are all in the front
    #
    var = False
    for i in range(len(shape))[::-1]:
        d = shape[i]
        if d is None:
            var = True
        elif var:
            shape[i] = None
            
    return dtype, shape
        
    


import json
attrs = {}
branches = {}

for k in sorted(tree.keys()):
    b=tree[k]
    interp = b.interpretation
    t=interp.type
    dtype, shape = type_shape(t)
    words = k.split("_",1)
    if len(shape)==1 and shape[0] is None and len(words)==2:
        bname, aname = words
        bdict = branches.setdefault(bname, {})
        bdict[aname] = dict(
            dtype=dtype.str,
            shape=shape[1:],
            source=k,
            subtype=None
        )
    else:
        attrs[k] = dict(
            dtype=dtype.str,
            shape=shape,
            source=k,
            subtype=None
        )
    print k, shape, dtype

schema = dict(attributes = attrs, branches=branches)
open(sys.argv[2],"w").write(json.dumps(schema, indent=4, sort_keys=True))




