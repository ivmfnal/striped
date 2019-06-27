import numpy as np
import zlib, base64, cPickle
import HPlot

class HistAxis(object):
   
    def __init__(self, type, name, sparse, label = None):
        self.Type = type
        self.Label = label
        self.Name = name
        self.IsSparse = sparse
        
    def __base_str__(self):
        return "type=%s, name=%s, starse=%s, label=%s" % (self.Type, self.Name, self.IsSparse, self.Label)

    @property
    def isSparse(self):
        return self.IsSparse 
        
    @property
    def label(self):
        return self.Label or self.Name

    @property
    def domain(self):
        return None
        
    def meta(self):
        d = dict(
            type = self.Type,
            name = self.Name,
            label = self.Label,
            is_sparse = self.IsSparse
        )
        d.update(self._specific_meta())
        return d
    
    @classmethod    
    def from_meta(cls, meta):
        type = meta["type"]
        if type == "linear":
            a = LinearAxis._from_meta(meta)
        elif type == "variable":
            a = VarAxis._from_meta(meta)
        elif type == "category":
            a = CatAxis._from_meta(meta)
        else:
            raise ValueError("Unknown axis type %s" % (type,))
        a.Type, a.Name, a.Label = type, meta["name"], meta["label"]
        return a
        
    @classmethod
    def _from_meta(cls, meta):
        raise NotImplementedError
        
    def _specific_meta(self):
        # override me
        return {}

class LinearAxis(HistAxis):
    
    def __init__(self, name, n, xmin, xmax, label = None):
        HistAxis.__init__(self, "linear", name, False, label)
        self.N = n
        self.XMin = float(xmin)
        self.XMax = float(xmax)
        bin = (self.XMax - self.XMin)/self.N
        self.Bins = self.XMin + np.arange(self.N+1)*bin
        self.Bin = bin
        
    def __str__(self):
        return "LinearAxis(%s, nbins=%s, xmin=%s, xmax=%s)" % (self.__base_str__(), self.N, self.XMin, self.XMax)

    @property
    def domain(self):
        return (self.XMin, self.XMax)

    def bins(self):
        return self.Bins

    def project(self, x):
        if x < self.XMin:   return -1
        elif x >= self.XMax:    return self.N
        else:
            return int((x-self.XMin)/self.Bin)
        
    def _specific_meta(self):
        return dict(
            xmin = self.XMin,
            xmax = self.XMax,
            bins = self.N            
        )

    @classmethod
    def _from_meta(cls, meta):
        return cls(meta["name"], meta["bins"], meta["xmin"], meta["xmax"])
        
class VarAxis(HistAxis):
    
    def __init__(self, name, bins, label = None):
        HistAxis.__init__(self, "variable", name, False, label)
        self.Bins = np.array(bins)
        self.XMin = bins[0]
        self.XMax = bins[-1]
        
    def __str__(self):
        return "LinearAxis(%s, bins=%s)" % (self.__base_str__(), self.Bins)

    @property
    def domain(self):
        return (self.XMin, self.XMax)

    def bins(self):
        return self.Bins
        
    def project(self, x):
        if x < self.XMin:   return -1
        elif x >= self.XMax:    return self.N
        else:
            for i, b in enumerate(self.Bins):
                if x <= b:
                    return i
            else:
                return self.N

    def _specific_meta(self):
        return dict(
            bins = self.Bins
        )

    @classmethod
    def _from_meta(cls, meta):
        return cls(meta["name"], meta["bins"])
        
class CatAxis(HistAxis):
    
    def __init__(self, name, label=None):
        HistAxis.__init__(self, "category", name, True, label)

    def __str__(self):
        return "CatAxis(%s)" % (self.__base_str__(),)

    def project(self, x):
        return x

    @classmethod
    def _from_meta(cls, meta):
        return cls(meta["name"])

def category(name, *params, **args):
    return CatAxis(name, *params, **args)

def bins(name, *params, **args):
    if len(params) == 3:
        return LinearAxis(name, params[0], params[1], params[2], label=args.get("label"))
    elif len(params) == 1:
        return VarAxis(name, params[0], label=args.get("label"))
    else:
        raise ValueError("bin accepts either 4 or 2 arguments")
        
class Hist:
    
    def __init__(self, *axes, **kv):
        self.AxesList = axes
        self.Axes = { x.Name:x for x in axes }
        self.NameList = sorted(self.Axes.keys())
        self.NameSet = set(self.NameList)
        self.DenseAxesList = sorted([name for name, x in self.Axes.items() if not x.isSparse])
        self.DenseBins = tuple([self.Axes[name].bins() for name in self.DenseAxesList])
        self.CountsShape = tuple(len(bins)-1 for bins in self.DenseBins)
        self.SparseAxesList = sorted([name for name, x in self.Axes.items() if x.isSparse])
        self.Weights = kv.get("weights")
        self.Constraints = {}
        self.clear()                    # this will initialize counts
        
    @staticmethod
    def build(axes, weights, counts):
        h = Hist(*axes, weights=weights)
        h.Counts = counts
        return h
        
    @property
    def id(self):
        return "Hist_%x" % (id(self),)
        
    def clear(self):
        # deallocate all counts arrays
        if len(self.SparseAxesList):
            self.Counts = {}
        else:
            self.Counts = np.zeros(self.CountsShape, dtype=np.float64)

    def zero(self):
        # zero all counts without deallocating anything
        if len(self.SparseAxesList):
            for k, c in self.Counts.items():
                c[...] = 0.0
            self.Counts = new_counts
        else:
            self.Counts[...] = 0.0
                    

    @property                        
    def fields(self):
        return self.NameList
        
    def fill(self, *params, **kv):
        # fill( { name: values, name: values} )
        # fill(name=values, name=values)
        
        args = {}
        
        if params:
            args.update(params[0])
        if kv:
            args.update(kv)
            
        # Check if all the axes are supplied
        assert len(self.NameSet.symmetric_difference(set(args.keys()))) == 0, "Invalid set of values provided"
        
        # Check array sizes
        l = None
        category_values = []
        numeric_values = []
        weights = None
        for name, value in sorted(args.items()):
            if name == self.Weights:
                weights = np.array(value)     # noop if it's already an array
                if l is None:   l = len(weights)
                assert l == len(weights), "All arrays must have the same length"
            else:
                axis = self.Axes[name]
                if axis.isSparse:
                    assert not isinstance(value, (np.ndarray, list, tuple)), "Sparse axis %s value must be scalar"
                    category_values.append(value)
                else:
                    array = np.array(value)     # noop if it's already an array
                    numeric_values.append(array)
                    if l is None:   l = len(array)
                    assert l == len(array), "All arrays must have the same length"
        category_values = tuple(category_values)        # to make it hashble
        
        # Do the fill
        if len(numeric_values):
            counts, _ = np.histogramdd(numeric_values, self.DenseBins, weights = weights)
        else:
            counts = 1.0 if weights is None else weights      # just a scalar 1, because all axes happen to be categorical

        if category_values:
            my_counts = self.Counts.get(category_values)
            if my_counts is None:
                self.Counts[category_values] = np.array(counts)
            else:
                my_counts += counts
        else:
            self.Counts += counts

    def counts(self):
        return self.Counts
        
    def add_counts(self, c1):
        assert isinstance(c1, dict) == isinstance(self.Counts, dict), "Histograms of different kinds can not be added"
        if isinstance(c1, dict):
            for k1, counts1 in c1.items():
                if k1 in self.Counts:
                    self.Counts[k1] += counts1
                else:
                    self.Counts[k1] = counts1.copy()
        else:
            self.Counts += c1
       
    def add(self, another):
        if isinstance(another, Hist):
                c1 = another.Counts
        elif isinstance(another, (int, float, np.ndarray, dict)):
                c1 = another
        self.add_counts(c1)
       
    def meta(self):
        return dict(
            axes = [self.Axes[name].meta() for name in self.NameList],
            weights = self.Weights
        )
        
    @staticmethod
    def empty_from_meta(meta):
        alist = [HistAxis.from_meta(m) for m in meta["axes"]]
        weights = meta["weights"]
        return Hist(*alist, weights=weights)

    @staticmethod 
    def serialize_array(counts):
        if isinstance(counts, np.ndarray):
            data = np.ascontiguousarray(counts).data
            data = base64.b64encode(zlib.compress(data, 1))
            return (counts.dtype.str, counts.shape, data)
        else:
            # assume scalar
            return counts
                
    @staticmethod 
    def deserialize_array(x):
        if isinstance(x, (tuple, list)):
            dtype, shape, data = x
            data = zlib.decompress(base64.b64decode(data))
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        else:
            return x
            
    def serialize_counts(self):
        if self.Counts is None:
            obj = None
        elif isinstance(self.Counts, dict):
            obj = {key:self.serialize_array(counts) for key, counts in self.Counts.items()}
        elif isinstance(self.Counts, np.ndarray):
            obj = self.serialize_array(self.Counts)
        return cPickle.dumps(obj)
    
    @staticmethod 
    def deserialize_counts(buffer):
        if not buffer:  return None
        obj = cPickle.loads(buffer)
        if isinstance(obj, dict):
            return {key:Hist.deserialize_array(counts) for key, counts in obj.items()}
        else:
            return Hist.deserialize_array(obj)
    
    def layerAsList(self, layer, constraints):
        if isinstance(layer, np.ndarray):
            lst = []
            flat_layer = layer.reshape((-1,))
            error_layer = np.sqrt(flat_layer)
            mask = (flat_layer-error_layer) <= 0.0
            error_layer[mask] = (flat_layer * 0.9)[mask]
            for i, (c, e) in enumerate(zip(flat_layer, error_layer)):
                indexes = np.unravel_index(i, layer.shape)
                dct = {"__count":float(c), "__error":e, "__hid":self.id}
                for iaxis, (idata, aname) in enumerate(zip(indexes, self.DenseAxesList)):
                    dct[aname] = self.DenseBins[iaxis][idata]
                    midbin = (self.DenseBins[iaxis][idata] + self.DenseBins[iaxis][idata+1])/2
                    dct["__midbin_" + aname] = midbin
                lst.append(dct)
        else:
            lst = [{"__count":float(layer), "__error":math.sqrt(float(layer)), "__hid":self.id}]
        return lst
        
    def countsAsList(self, constraints):
        if self.Counts is None: return []
        elif isinstance(self.Counts, dict):
            out = []
            sparseNames = self.SparseAxesList
            for keys, layer in self.Counts.items():
                keys_dict = {name:value for name, value in zip(sparseNames, keys)}
                counts = self.layerAsList(layer, constraints)
                #print keys, layer, counts
                for dct in counts:
                    dct.update(keys_dict)
                    out.append(dct)
            return out
        else:
            return self.layerAsList(self.Counts, constraints)
            
    def select_old(self, **constraints):
        name, value = constraints.items()[0]
        axis = self.Axes[name]
        new_axes = [ax for ax in self.AxesList if ax.Name != name]
        counts = None
        if axis.isSparse:
            if len(self.SparseAxesList) == 1:
                counts = self.Counts.get((value,))
            else:
                counts = {}
                i = self.SparseAxesList.index(name)
                for keys, layer in self.Counts.items():
                    keys = tuple(keys[:i]+keys[i+1:])
                    counts[keys] = layer
        else:
            # the constrained axis is dense
            ibin = axis.project(value)
            iaxis = self.DenseAxesList.index(name)
            ndense = len(self.DenseAxesList)
            selector = [slice(None, None, None)]*ndense
            selector = selector[:iaxis] + [ibin] + selector[iaxis+1:]
            if isinstance(self.Counts, dict):
                counts = {}
                for keys, layer in self.Counts.items():
                    counts[keys] = layer[selector]
            else:
                counts = self.Counts[selector]
        h = Hist.build(new_axes, self.Weights, counts)
        remaining = {n:v for n, v in constraints.items() if n != name}
        if remaining:
            h = self.slice(**remaining)
        return h
        
    def select(self, **constraints):
        remaining_axes = [ax for ax in self.AxesList if not ax.Name in constraints]
        new_hist = Hist(*remaining_axes, weights = self.Weights)
        new_dense = new_hist.DenseAxesList
        new_sparse = new_hist.SparseAxesList
        
        dense_index_sel = []
        for axis_name in self.DenseAxesList:
            ibin = None
            if axis_name in constraints:
                axis = self.Axes[axis_name]
                ibin = axis.project(constraints[axis_name])
            dense_index_sel.append(ibin)

        dense_index_sel = tuple(dense_index_sel) if dense_index_sel else None
        sparse_axes_indexes = tuple(i
                for i, axis_name in enumerate(self.SparseAxesList)
                if axis_name in constraints)
        sparse_axes_values = tuple(constraints[axis_name]
                for axis_name in self.SparseAxesList
                if axis_name in constraints)

        new_counts = self.Counts 
        
        # project sparse
        if isinstance(my_counts, dict):
        
            if not new_sparse:
                # all sparse axes are included in the constraints
                new_counts = self.Counts.get(sparse_axes_values)
            else:
                new_counts = {}
                for keys, counts in self.Counts.items():
                    selected_key_values = tuple(keys[i] for i in sparse_axes_indexes)
                    if selected_key_values == sparse_axes_values:
                        remaining_key_values = tuple(key_value for i, key_value in enumerate(keys)
                                if not i in sparse_axes_indexes)
                        new_counts[remaining_key_values] = counts

        # project dense
        if isinstance(new_counts, dict):
            new_counts = { keys : counts[dense_index_sel] for keys, counts in new_counts.items() }
        elif isinstance(new_counts, np.ndarray):
            new_counts = new_counts[dense_index_sel]
        else:
            new_counts = new_counts or 0.0
        new_hist.Counts = new_counts
        return new_hist
        
        
    def project(self, *axis_names):
        remaining_axes = [ax for ax in self.AxesList if not ax.Name in axis_names]
        axis_names_set = set(axis_names)
        new_hist = Hist(*remaining_axes, weights = self.Weights)
        new_dense = new_hist.DenseAxesList
        new_sparse = new_hist.SparseAxesList

        dense_axis_index_list = tuple(i for i, axis_name in enumerate(self.DenseAxesList)
                        if axis_name in axis_names_set)

        new_counts_shape = None if not new_dense \
                else tuple(n for n in self.CountsShape if not n in dense_axis_index_list)


        sparse_axes_indexes = tuple(i
                for i, axis_name in enumerate(self.SparseAxesList)
                if axis_name in constraints)
        sparse_axes_values = tuple(constraints[axis_name]
                for axis_name in self.SparseAxesList
                if axis_name in constraints)

        new_counts = {} if new_sparse else (np.zeros(new_hist.CountsShape, dtype=np.float64) if new_dense else 0.0)   
        
        # project dense component
        if isinstance(self.Counts, dict):
            for keys, counts in self.Counts.items():
                if isinstance(counts, np.ndarray):
                    counts = np.sum(counts, axis=dense_axis_index_list)
                new_counts[keys] = counts
        elif isinstance(self.Counts, np.ndarray):
            new_counts = np.sum(self.Counts, axis=dense_axis_index_list)
        else:
            new_counts = self.Counts
        
        # project sparse
        if isinstance(new_counts, dict) and sparse_axes_indexes:
            # FIXME !
            pass
            
                

        
        
        new_counts = self.Counts
        
        if isinstance(self.Counts, np.ndarray):
            new_counts = np.sum(self.Counts, axis=dense_axis_index_list)
        elif isinstanxce(self.Counts, dict):
            new_counts = {}
            for keys, counts in self.Counts.items():
                if dense_axis_index_list:
                    counts = np.sum(counts, axis = dense_axis_index_list)
                selected_key_values = tuple(keys[i] for i in sparse_axes_indexes)
                if selected_key_values == sparse_axes_values:
                    remaining_key_values = tuple(key_value for i, key_value in enumerate(keys)
                            if not i in sparse_axes_indexes)
                    if remaining_key_values:
                        c = new_counts.setdefault(remaining_key_values, np.zeros(new_counts_shape, dtype=np.float64))
                        
                    
                
        
        # project sparse
        if isinstance(my_counts, dict):
        
            if not new_sparse:
                # all sparse axes are included in the constraints
                new_counts = self.Counts.get(sparse_axes_values)
            else:
                new_counts = {}
                for keys, counts in self.Counts.items():
                    selected_key_values = tuple(keys[i] for i in sparse_axes_indexes)
                    if selected_key_values == sparse_axes_values:
                        remaining_key_values = tuple(key_value for i, key_value in enumerate(keys)
                                if not i in sparse_axes_indexes)
                        new_counts[remaining_key_values] = counts

        # project dense
        if isinstance(new_counts, dict):
            new_counts = { keys : counts[dense_index_sel] for keys, counts in new_counts.items() }
        elif isinstance(new_counts, np.ndarray):
            new_counts = new_counts[dense_index_sel]
        else:
            new_counts = new_counts or 0.0
        new_hist.Counts = new_counts
        return new_hist
                
#
# Splitting
#

    def beside(self, axis_name):
        # for now, require that the axis is sparse
        assert axis_name in self.Axes and self.Axes[axis_name].isSparse, "The axis to spread over must be sparse"
        return HPlot.Spread(self, haxis = axis_name)
        
    def below(self, axis_name):
        # for now, require that the axis is sparse
        assert axis_name in self.Axes and self.Axes[axis_name].isSparse, "The axis to spread over must be sparse"
        return HPlot.Spread(self, vaxis = axis_name)
        
    def split(self, row=None, column=None):
        assert not (row is None and column is None)
        assert row is None or row in self.Axes and self.Axes[row].isSparse, "The axis to split over must be sparse"
        assert column is None or column in self.Axes and self.Axes[column].isSparse, "The axis to split over must be sparse"
        return HPlot.Split(self, row=row, column=column)
        
        
#
# Plotting
#

    def line(self, axis_name, **args):
        assert axis_name in self.Axes
        return HPlot.lineChart(self, axis_name, split=None, **args)
        
    def area(self, axis_name, **args):
        assert axis_name in self.Axes
        return HPlot.areaChart(self, axis_name, split=None, **args)
        
    def step(self, axis_name, **args):
        assert axis_name in self.Axes
        return HPlot.stepChart(self, axis_name, split=None, **args)
        
    def heatmap(self, xaxis, yaxis, **args):
        assert xaxis in self.Axes
        assert yaxis in self.Axes
        return HPlot.heatmapChart(self, xaxis, yaxis, split=None, **args)
        
#
# View interface
#
    def collect_data(self, data, mapping):
        hid = self.id
        counts = self.countsAsList(self.Constraints)
        #datasets[hid] = counts
        data[self.id] = counts

    def vegalite(self, data, mapping, inout = {}):
        return
        
            
if __name__ == '__main__':
    import pprint, json
    h3 = Hist(bin('x', 100,0,1), bin('y', [0.0, 0.1, 0.5, 0.7]), category('c'))
    x = np.random.random((100,))
    y = np.random.random((100,))
    h3.fill(x=x, y=y, c="hello")
    x = np.random.random((100,))
    y = np.random.random((100,))
    h3.fill(x=x, y=y, c="world")
    
    h2 = Hist(bin('x', 10,0,1), category('c', label="category"))
    x = np.random.random((100,))
    h2.fill(x=x, c="hello")
    x = np.random.random((100,))
    h2.fill(x=x, c="world")
    print h2.Counts
    #print h2.meta()
    
    plot = h2.area(color="c")
    print json.dumps(plot.vegalite(), indent=4)
    
    
        
        
        
            
    
