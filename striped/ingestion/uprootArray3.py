import uproot, itertools, time
from uproot.interp.auto import interpret
import numpy as np
import re


#
# Deal with uproot2/3 incompatibilities
#

try:
        # uproot 2
        genobjclass = uproot.interp.jagged.asobjs
except:
        # uproot 3
        genobjclass = uproot.interp.objects.asgenobj

try:
        # uproot 3
        stringobjclass = uproot.interp.objects.asstring
except:
        # uproot 2
        stringobjclass = uproot.interp.strings.asstrings        





class UprootArrayBase(object):

    # Abstract class dening the interface required by the ingestion tools
    
    def __init__(self, dtype, shape):
        self.DType = dtype
        if shape:
            shape = tuple([None if not x else x for x in shape])
        self.Shape = shape
        
    def stripesAndSizes(self, gsizes):
        raise NotImplementedError 

    def stripes(self, gsizes):
        for d, s in self.stripesAndSizes(gsizes):
            yield d
        
    def stripeSizes(self, gsizes):
        for d, s in self.stripesAndSizes(gsizes):
            yield s

    def hasSizes(self):
        return False
    
    def splitShape(self):
        nv = 0
        fixed = ()
        for i, dim in enumerate(self.Shape):
            if dim is None:
                nv += 1
            else:
                return nv, self.Shape[i:]
        return nv, fixed
                   


class UprootArray(UprootArrayBase):

    def __init__(self, branch, interp, dtype, shape, name, array = None, counter = None,
                subtype = None):
                
        UprootArrayBase.__init__(self, dtype, shape)
        # "UprootArray constructor called"
        self.Interp = interp
        self.Branch = branch
        self.Counter = counter
        self.Name = name
        self.Array = array
        self.Subtype = subtype

    @staticmethod
    def fromSimple(dtype, shape, data):
        if shape and shape[0] is None:
            return UprootArray_Simple2D(data, dtype, shape)
        else:
            return UprootArray_Simple1D(data, dtype, shape)
        
    @staticmethod
    def create(branch, dtype = None, shape = None):
    
        # shape does not include the event dimension
    
        if isinstance(branch, (list, tuple, np.ndarray)):
            if shape:
                return UprootArray_Simple2D(branch, dtype, shape)
            else:
                return UprootArray_Simple1D(branch, dtype, shape)
            
        interp = branch.interpretation
        #print "interp:", type(interp)
        if isinstance(interp, genobjclass):
            #print "asobjs"
            return UprootArray._createFromObjects(branch, interp)
            
        if isinstance(interp, stringobjclass):
            return UrootArray_Strings(branch, interp)
            
        elif isinstance(interp, uproot.interp.jagged.asvar):
            #print "asvar"
            a = branch.array()
            #print type(a)
            if isinstance(a, uproot.interp.strings.ListStrings):
                return UprootArray_ListStrings(branch, interp, a)
                
            elif isinstance(a, uproot.interp.jagged.JaggedJaggedArray):
                return UprootArray_JaggedJagged(branch, interp, a)
                
        elif isinstance(interp, uproot.interp.jagged.asjagged):
            return UprootArray_Jagged(branch, interp)
            
        elif isinstance(interp, uproot.interp.numerical.asdtype):
            return UprootArray_Flat(branch, interp)
            
        else:
            raise ValueError("UprootArray.create(): Unknown type of interpretation %s for branch: %s %s" % (
                type(interp), branch.name, type(branch)))

    IdentParseRE = re.compile("[a-zA-Z]\w*\((?P<class>[a-zA-Z]\w*)(,.*)?\)")

    @staticmethod
    def _createFromObjects(branch, interp):
        #print "_createFromObject"
        ident = interp.identifier
        match = UprootArray.IdentParseRE.match(ident)
        if match:
            class_name = match.group("class")
            #print class_name
            if not class_name in _ObjectClasses:
                raise ValueErorr("Unknown object class name %s for branch %s. Identifier: '%s'" % 
                        (class_name, branch.name, ident))
            shape = interp.dtype.shape
            if shape[0] == 0:
                return _ObjectClasses[class_name]["2D"](branch, interp)
            else:
                return _ObjectClasses[class_name]["1D"](branch, interp) 
        else:
            raise ValueError("Can not parse identity '%s' for branch %s" % (ident, branch.name))
                
    @property
    def array(self):
        if self.Array is None:
            self.Array = self.Branch.array()
        return self.Array

    @property
    def counter(self):
        if self.Counter is None:
            self.Counter = self.getCounter()
        return self.Counter
            
    def getCounter(self):
        if self.Branch is None: return None
        counter_name = self.Branch.fLeaves[0].fLeafCount
        if counter_name is None:
                return None
        return counter_name.fName.rstrip("_")

    def flatArray(self):
        raise NotImplementedError

    def flattenedShape(self):
        return self.Shape
        
        

class Uproot1DArray(UprootArray):

    def stripes(self, groups):
        for a, s in self.stripesAndSizes(groups):
            yield a
            
    def stripesAndSizes(self, groups):
        ja = 0
        arr = self.flatArray()
        for n in groups:
            yield arr[ja:ja+n], None
            ja += n

class Uproot2DArray(UprootArray):

    def stripes(self, groups):
        arr = self.flatArray()
        i = 0
        for g in groups:
            #print self.__class__.__name__
            #print "arr[i]", arr[i]
            #print "arr[i+1]", arr[i+1]
            #print "arr[i+2]", arr[i+2]
            parts = [p for p in arr[i:i+g] if len(p) > 0]
            out = np.concatenate(parts) if len(parts) > 0 else np.array([], dtype=self.DType)
            yield out
            i += g
            
    def stripeSizes(self, groups):
        arr = self.array
        i = 0
        try:    lens = np.array(list(map(len, arr)))
        except:
            print("Can not calculate length in:", arr)
            raise
        for g in groups:
            yield lens[i:i+g]
            i += g

    def stripesAndSizes(self, groups, dtype = None):
        dtype = dtype or self.DType
        arr = self.flatArray()
        i = 0
        size_array = np.array(list(map(len, arr)))
        for g in groups:
            segment = arr[i:i+g]
            concatenated = []
            for lst in segment:
                if len(lst):
                    concatenated.extend(list(lst))
            if len(concatenated) == 0:
                stripe = np.array([], dtype=dtype)
            else:
                stripe = np.array(concatenated)
            yield stripe, size_array[i:i+g]
            i += g


    def hasSizes(self):
        return True
        
    def flattenedShape(self):
        j = len(self.Shape)
        for i in range(len(self.Shape)):
            if i is not None:
                j = i
                break
        return (None,) + self.Shape[j:]
        
class Uproot1D_TLorentzVector(Uproot1DArray):
    def __init__(self, branch, interp):
        Uproot1DArray.__init__(self, branch, interp, 'O', interp.dtype.shape, branch.name,
            subtype = "TLorentzVector")
        assert interp.dtype[0].str == 'O'
        
    def flatArray(self):
        lst = [x.tolist() for x in self.array]
        return np.array(lst, dtype=np.float64)
        

class Uproot2D_TLorentzVector(Uproot2DArray):

    def __init__(self, branch, interp):
        #print "Uproot2D_TLorentzVector constructor called"
        Uproot2DArray.__init__(self, branch, interp, 'O', interp.dtype.shape, branch.name,
            subtype = "TLorentzVector")
            
        #raise NotImplementedError("Uproot2D_TLorentzVector class not tested")
        assert interp.dtype.subdtype[0] == np.object
        
        
    def stripesAndSizes(self, groups):
        #t0 = time.time()
        as_tuples = self.Branch.array(
            uproot.interp.asjagged(
                uproot.interp.asdtype(
                    {"names": ["x", "y", "z", "t"], 
                     "formats": [">f8", ">f8", ">f8", ">f8"], 
                     "offsets": [32, 32+8, 32+16, 32+24]}), 
                     skip_bytes=10))
        sizes = np.array([len(x) for x in as_tuples])                     
        as_array = self.Branch.array(
                uproot.interp.asjagged(
                    uproot.interp.asdtype(">f8", todims=(8,)), 
                    skip_bytes=10)).content[:, 4:]
        #t1 = time.time()
        #print "Uproot2D_TLorentzVector: decoding time: %f" % (t1 - t0)
                    
        ievent = 0
        ipoint = 0
        for n in groups:
            s = sizes[ievent:ievent+n]
            nvectors = np.sum(s)
            yield as_array[ipoint:ipoint+nvectors].copy(), s
            ipoint += nvectors
            ievent += n

    def stripes(self, groups):
        for stripe, size in self.stripesAndSizes(groups):
            yield stripe

    def stripeSizes(self, groups):
        for stripe, size in self.stripesAndSizes(groups):
            yield size

        
    def flatArray(self):
        raise NotImplementedError("Do we actually need this?")
#        lst = []
#        for vector_list in self.array:
#            tuples = np.array([v.tolist() for v in vector_list], dtype=np.float64)
#            tuples = np.array([(v.x, v.y, v.z, v.t) for v in vector_list], dtype=np.float64)
#            lst.append(tuples)
#        return lst
            
        return [np.array([(v.x, v.y, v.z, v.t) for v in vector_list], dtype=np.float64) 
                    for vector_list in self.array
            ]

class UprootArray_Simple1D(Uproot1DArray):

    def __init__(self, lst, dtype, shape):
        Uproot1DArray.__init__(self, None, None, dtype, shape, None, array=lst)
        assert dtype is not None
        assert shape is not None

    def flatArray(self):
        return self.array
        
class UrootArray_Strings(Uproot1DArray):
    def __init__(self, branch, interp):
        Uproot1DArray.__init__(self, branch, interp, 'S', [], branch.name, array=branch.array())
        
    def flatArray(self):
        return np.array(self.Array)
        
class UprootArray_Simple2D(Uproot2DArray):

    def __init__(self, lst, dtype, shape):
        UprootArray.__init__(self, None, None, dtype, shape, None, array=lst)
        assert dtype is not None
        assert shape is not None

    def flatArray(self):
        return self.array
        
        
class UprootArray_Flat(Uproot1DArray):

    def __init__(self, branch, interp):
        Uproot1DArray.__init__(self, branch, interp,
                interp.todtype.str, interp.todims, branch.name)

    def flatArray(self):
        return self.array
        
class UprootArray_Jagged(Uproot2DArray):
    def __init__(self, branch, interp):
        #print "UprootArray_Jagged", branch.name
        Uproot2DArray.__init__(self, branch, interp, 
                interp.asdtype.todtype.str, (None,) + interp.asdtype.todims, branch.name)

    def flatArray(self):
        return self.array
    
    def stripes(self, groups):
        for stripe, size in self.stripesAndSizes(groups):
            yield stripe

    def stripeSizes(self, groups):
        for stripe, size in self.stripesAndSizes(groups):
            yield size

    def stripesAndSizes(self, groups):
        arr = self.array
        contents = arr.content
        starts = arr.starts
        stops = arr.stops
        sizes = arr.stops - arr.starts
        js = 0
        for igroup, n in enumerate(groups):
            make_view = np.all(stops[js:js+n-1] == starts[js+1:js+n])
            #print "%s: make_view: %s" % (self.Branch.Name, make_view)
            if make_view:
                yield contents[starts[js]:stops[js+n-1]], sizes[js:js+n]
            else:
                # make copy
                yield np.concatenate([contents[starts[js+i]:stops[js+i]] for i in range(n)]), sizes[js:js+n]
            js += n

            
class UprootArray_ListStrings(Uproot2DArray):

    def __init__(self, branch, interp, array):
        UprootArray.__init__(self, branch, interp, 
            'S', (None,) + interp.asdtype.todims, 
            branch.name, array=array)

    def flatArray(self):
        out = [np.array(event_data) for event_data in self.array]
        #print out
        return out
        
    def stripesAndSizes(self, groups):
        return Uproot2DArray.stripesAndSizes(self, groups, '|S1')

class UprootArray_Strings(Uproot1DArray):

    def __init__(self, branch, interp):
        UprootArray.__init__(self, branch, interp, 
            'S', (None,), 
            branch.name, branch.array())

    def flatArray(self):
        out = np.array(self.array)
        #print out
        return out
        
class UprootArray_JaggedJagged(Uproot2DArray):

    def __init__(self, branch, interp, array):
        UprootArray.__init__(self, branch, interp, 
            array.fromdtype.str, (None,None) + interp.asdtype.todims, 
            branch.name, array=array)
            
    def flatArray(self):
        return [np.concatenate(e) for e in self.array]
        
_ObjectClasses = {
    "TLorentzVector":   {
        "1D":   Uproot1D_TLorentzVector,
        "2D":   Uproot2D_TLorentzVector
    }
}
     

if __name__ == '__main__':
        import sys

        f = uproot.open(sys.argv[1])
        t = f[sys.argv[2]]

        ua = UprootArray.create(t["Jets"])
        for stripe, sizes in ua.stripesAndSizes([10,10,10]):
            print(sizes)
            print(type(stripe), stripe)
            
                                       

