import math

class View(object):
    
    def __init__(self, parent=None):        
        # parent is either link to the previous view in the chain, or a link to the underlying histogram or None
        # For example:
        # below( h1.step(), h2.step() ) has Parent = None
        # h1.below("c").area("x") has Parent = h1.below("c")
        # 
        self.Parent = parent
        
    @property
    def hist(self):
        # recursively, get the underlying histogram, if any
        p = self.Parent
        if isinstance(p, View):
            p = p.hist
        return p

    def vegalite(self, data, mapping, inout):
        raise NotImplementedError
        
        
class Plot(View):
    # something that can be rendered to a canvas
    # for example:
    # hist.below(x).area(c)
    # besides(h1.step(x), h2.area(y))
    #
    

    VegaHeader = {"$schema": "https://vega.github.io/schema/vega-lite/v2.json"}

    def plot(self):
        data, mapping = {}, {}
        self.collect_data(data, mapping)
        inout = {}
        inout.update(self.VegaHeader)
        inout["datasets"] = data
        self.vegalite(data, mapping, inout)
        #print inout
        return inout
        
    def to(self, output):
        return output(self.plot())
        
#
# Arrangements
#

class Arrangement(Plot):      # placement of plots
    
    def __init__(self, plots):
        Plot.__init__(self)
        self.Plots = plots      # iterable
        for p in plots:
            assert isinstance(p, Plot), "Arrangement has to be a collection of plotables"
            
    def collect_data(self, data, mapping):
        for p in self.Plots:
            p.collect_data(data, mapping)
    
class BesideArrangement(Arrangement):
    
    def vegalite(self, data, mapping, inout = {}):
        inout["hconcat"] = [v.vegalite(data, mapping, {}) for v in self.Plots]
        return inout
    
class BelowArrangement(Arrangement):

    def vegalite(self, data, mapping, inout = {}):
        inout["vconcat"] = [v.vegalite(data, mapping, {}) for v in self.Plots]
        return inout
    
class GridArrangement(Arrangement):
    def __init__(self, width, plots):
        Placement.__init__(self, plots)
        self.Width = width
   
    def vegalite(self, data, mapping, inout = {}):
        vlst = []
        hlst = []
        for v in self.Plots:
            if len(hlst) >= self.Width:
                vlst.append(hlst)
                hlst = []
            hlst.append(v.vegalite(data, mapping))
        if hlst:
            vlst.append(hlst)
        inout["vconcat"] = [
                {
                    "hconcat":hlst
                }
                for hlst in vlst
            ]
        return inout
        
def below(*plots):
    return BelowArrangement(plots)
    
def beside(*plots):
    return BesideArrangement(plots)
    
def grid(width, *plots):
    return GridArrangement(width, plots)
    
        
#
# Spread
# example: h.below("axis")
#

class Split(View):
    def __init__(self, hist, row = None, column = None):
        View.__init__(self, hist)
        self.H = hist
        self.RAxis = row
        self.CAxis = column
        
    #def collect_data(self, data, mapping):
    #    self.H.collect_data(data, mapping)

    def vegalite(self, data, mapping, inout = {}):
        enc = inout.setdefault("encoding", {})
        
        if self.RAxis:
            raxis = self.H.Axes[self.RAxis]
            enc["row"] = {
                    "field":    self.RAxis,
                    "header":   {   "title":    raxis.label  },
                    "type":     "nominal"
                }
        if self.CAxis:
            caxis = self.H.Axes[self.CAxis]
            enc["column"] = {
                    "field":    self.CAxis,
                    "header":   {   "title":    caxis.label  },
                    "type":     "nominal"
                }
        return inout
#
# Charts
#

    def line(self, xaxis, **chart_args):
        return Chart(self.H, xaxis, LineMarker(), split=self, **chart_args)
        
    def area(self, xaxis, **chart_args):
        return Chart(self.H, xaxis, AreaMarker(), split=self, **chart_args)
        
    def step(self, xaxis, **chart_args):
        return Chart(self.H, xaxis, StepMarker(), split=self, **chart_args)
        
    def heatmap(self, xaxis, yaxis, **chart_args):
        return Chart2D(self.H, xaxis, yaxis, split=self, **chart_args)

#
# Chart
# E.g.:
# h.below(x).area(c)
#

class Chart2D(Plot):
    
    def __init__(self, hist, xaxis, yaxis, split=None, color_schema=None):
        self.H = hist
        self.Split = split
        self.XAxis = xaxis
        self.YAxis = yaxis
        self.ColorSchema = color_schema     # use Vega default
        
    def collect_data(self, data, mapping):
        self.H.collect_data(data, mapping)
    
    def vegalite(self, data, mapping, inout = {}):
        hist = self.H
        xaxis = hist.Axes[self.XAxis]
        yaxis = hist.Axes[self.YAxis]
        
        encoding = {
            "x":    {
                'axis': {'title': xaxis.label},
                'field': self.XAxis,
                'scale': {'zero': False},
                'type': 'quantitative'
           },
           "y": {
                'axis': {'title': yaxis.label},
                'field': self.YAxis,
                'scale': {'zero': False},
                'type': 'quantitative'
           },
           "color": {
                'field': '__count',
                'legend': {'title': 'entries per bin'},
                'scale': {'zero': True},
                'type': 'quantitative'
           }
        }
        
        if not xaxis.isSparse:
            encoding["x"]["bin"] = {"extent": [xaxis.XMin, xaxis.XMax]}
        
        if not yaxis.isSparse:
            encoding["y"]["bin"] = {"extent": [yaxis.XMin, yaxis.XMax]}
        
        mark = "rect"

        if self.ColorSchema is not None:
            config = inout.setdefault("config", {})
            config_range = config.setdefault("range", {})
            config_range["heatmap"] = {"scheme":self.ColorSchema}
            #print inout["config"]
        
        inout["encoding"] = encoding
        inout["mark"] = mark
        inout["data"] = {"name":self.H.id}
        return inout
        

class Chart(Plot):
    
    def __init__(self, hist, xaxis, marker, split=None, color=None, stack=True, yscale=None, xscale=None, error=False):
        assert not (stack and error), "Can not draw errors for stacked chart"
        assert not (stack and yscale=="log"), "Can not draw stacked chart with logscale"
        self.H = hist
        self.Split = split
        self.XAxis = xaxis
        self.CAxis = color
        self.Stack = stack
        self.XScale = xscale
        self.YScale = yscale
        self.Errors = error
        self.Marker = marker
        
    def collect_data(self, data, mapping):
        self.H.collect_data(data, mapping)
        
    def x_parts(self, midbin=False):
        xaxis = self.H.Axes[self.XAxis]
        transforms = []
        encodings = {
            "x":    {   
                'axis': {'title': xaxis.label},
                'field': self.XAxis if not midbin 
				else "__midbin_"+self.XAxis,
                'scale': {'zero': False, 'domain':xaxis.domain},
                'type': 'quantitative'
            }
        }
        if self.XScale: 
            encodings["x"]["scale"] = {"type":self.XScale}
            if self.XScale == "log":
                transforms = [{"filter":   {"field":   self.XAxis, "gt": 0.0}}]
        return encodings, transforms

    def data_layer(self, data, inp):
        encodings, transforms = self.x_parts(midbin = self.Marker.xoffset=="mid")
        hist = self.H
        encodings["y"] = {
                'axis': {'title': "Entries per bin"},
                'field': '__count',
                'type': 'quantitative'
            }
        if self.YScale: 
            encodings["y"]["scale"] = {"type":self.YScale}
            if self.YScale == "log":
                transforms += [
                    {"filter":   "datum.__count - datum.__error > 0"},
                    {"filter":   {"field":   "__count", "gt": 0.0}}
                ]
        if self.Stack:
            encodings["y"]["aggregate"] = "sum"
        mark = self.Marker.point_mark
        return encodings, transforms, mark

    def errors_layer(self, data, inp):
        encodings, transforms = self.x_parts(midbin=True)
        hist = self.H
        encodings.update(
            {
                "y":{
                    'field': '__count_minus_error',
                    'type': 'quantitative'
                },
                "y2":{
                    'field': '__count_plus_error',
                    'type': 'quantitative'
                }
            }
        )
        
        transforms += [
            {'as': '__count_minus_error', 'calculate': 'datum.__count - datum.__error'},
            {'as': '__count_plus_error', 'calculate': 'datum.__count + datum.__error'}
        ]
            
        if self.YScale: 
            encodings["y"]["scale"] = {"type":self.YScale}
            if self.YScale == "log":
                transforms += [
                    {"filter":   "datum.__count - datum.__error > 0"},
                    {"filter":   {"field":   "__count", "gt": 0.0}}
                ]
                
        if self.Stack:
            encodings["y"]["aggregate"] = "sum"
        mark = self.Marker.error_mark
        return encodings, transforms, mark

        

    def vegalite(self, data, mapping, inout = {}):
        hist = self.H
        xaxis = hist.Axes[self.XAxis]
        if self.Split is not None:
            self.Split.vegalite(data, mapping, inout)
        

        data_encodings, data_transform, data_mark = self.data_layer(data, inout)
        if self.CAxis:
            caxis = self.H.Axes[self.CAxis]
            data_encodings["color"] = {
                "field":    self.CAxis,
                "legend":   {   "title":    caxis.label },
                "type": "nominal"
            }
        if self.Stack:
            data_encodings["y"]["aggregate"] = "sum"         # do not worry about errors here
            
        if self.Errors:
            errors_encodings, errors_transform, errors_mark = self.errors_layer(data, inout)
            layers = [
                {   # data
                    "encoding":     data_encodings,
                    "mark":         data_mark,
                    "transform":    data_transform
                },
                {   # errors
                    "encoding":     errors_encodings,
                    "mark":         errors_mark,
                    "transform":    errors_transform
                },
            ]
            inout["layer"] = layers
        else:
            inout.update({
                    "encoding":     data_encodings,
                    "mark":         data_mark,
                    "transform":    data_transform
                })
        inout["data"] = {"name":self.H.id}
        return inout

class Marker(object):
	error_mark = "rule"    
	xoffset = None

class AreaMarker(Marker):
    
    point_mark = {'clip': True, 'interpolate': 'step-before', 'type': 'area'}
        
class StepMarker(Marker):
    
    point_mark = {'clip': True, 'interpolate': 'step-before', 'type': 'line'}
        
class LineMarker(Marker):
    
    point_mark = {'clip': True, 'type': 'line'}
    xoffset = "mid"
        
def stepChart(hist, axis_name, **args):
    return Chart(hist, axis_name, StepMarker(), **args)

def lineChart(hist, axis_name, **args):
    return Chart(hist, axis_name, LineMarker(), **args)

def areaChart(hist, axis_name, **args):
    return Chart(hist, axis_name, AreaMarker(), **args)

def heatmapChart(hist, xaxis, yaxis, **args):
    return Chart2D(hist, xaxis, yaxis, **args)
