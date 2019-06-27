import uproot, numpy as np, json, sys

fname = sys.argv[1]


f=uproot.open(fname)
tree=f["Events"]


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
        


attrs, branches = {}, {}
for k, tb in sorted(tree.items()):
    ti = tb.interpretation
    bname, bdesc = None, None
    if ti is not None:
        bname = k
        bdesc = {}
        #print "----- Branch %s -----" % (bname,)
    else:
        bdesc = attrs
        #print "----- Attributes -----"
    for sk, sb in sorted(tb.items()):
        i = sb.interpretation
        t = i.type
        try:    
            dtype, shape = type_shape(t)
            dtype = dtype.str
        except: 
            if sk.endswith(".pfCands") or sk.endswith(".svtx"):
                dtype, shape = ">i2", [None, None]
            else:
                print '==== can not get dtype/shape for %s' % (sk,)
                print "    ", type(i), t
                print "type:", dir(t)
                print "interpretation:", dir(i)
                dtype, shape = None, None
        #print sk, dtype, shape

        if dtype[0] == ">":
            dtype = "<" + dtype[1:]
        
        if bname is None:
            # event attr
            aname = sk
            attrs[aname] = dict(
                dtype=dtype,
                shape = shape,
                source = sk,
                subtype = None
            )
        else:
            # branch attr
            assert shape[0] == None
            assert sk.startswith(bname+".")
            _,aname = sk.split(".")
            shape = shape[1:]
            if shape and shape[0] is None:
                bdesc[aname+"__count"] = dict(
                    dtype = "<u2",
                    shape = [],
                    source = ":count_of:"+sk,
                    subtype = None
                )
            bdesc[aname] = dict(
                dtype = dtype,
                shape = [],
                source = sk,
                subtype = None
            )
    if bname is not None:
        branches[bname] = bdesc

schema = {"attributes": attrs, "branches": branches}
print json.dumps(schema, indent=4, sort_keys=True)

            

