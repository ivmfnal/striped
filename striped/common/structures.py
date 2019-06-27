import numpy as np


def stripe_key(dataset_name, column_name, rgid):
    return "%s:%s:%d.bin" % (dataset_name, column_name, rgid)

def rginfo_key(dataset_name, rgid):
    return "%s:@@rginfo:%d.json" % (dataset_name, rgid)

def cdesc_key(dataset, cname):
    return "%s:%s:@@desc.json" % (dataset, cname)
    
def standalone_data_key(dataset_name, key, json=False):
    ext = ".json" if json else ".bin"
    if dataset_name:
        return "%s:@@data:%s.%s" % (dataset_name, key, ext)
    else:
        return "@@data:%s.%s" % (dataset_name, key, ext)
    
StripeHeaderFormatVersion = "1.0"

def stripe_header(array):
    return "#__header:version=%s;dtype=%s#" % (StripeHeaderFormatVersion, array.dtype.str)
    
def data_header(data):
    if isinstance(data, np.ndarray):
        return "#__header:version=%s;dtype=%s;shape=%s#" % (StripeHeaderFormatVersion, data.dtype.str, data.shape)
    else:
        return "#__header:version=%s;dtype=bytes#" % (StripeHeaderFormatVersion,)

def serialize_array(arr):
	return bytes(data_header(arr)) + array.data

def deserialize_array(text):
	assert text.startswith("#__header:")
	return parse_data(text)
	

def format_array(array):
    return bytes(stripe_header(array)) + bytes(array.data)

def parse_data(data):
    if not data: return None
    if data[:10] == "#__header:":
        iend = data.find("#", 10)
        if iend >= 0:
            hdr = data[10:iend]
            data = data[iend+1:]
        parts = hdr.split(";")
        shape = (-1,)
        dtype = "bytes"
        for part in parts:
            name, value = part.split("=",1)
            if name == "shape":
                value = value[1:-1]     # remove outer parenthesis
                shape = tuple([int(x) for x in value.split(",")])
            elif name == "dtype":
                dtype = value
        if dtype != "bytes":
            data = np.frombuffer(data, dtype).reshape(shape)
    return data

def standalone_data_key(key, dataset_name=None, json=False):
    dataset_name = dataset_name or "*"
    ext = ".json" if json else ".bin"
    return "%s:@@data:%s%s" % (dataset_name or "", key, ext)
    

    


    
class ColumnDescriptor:

    Version = 1.0

    def __init__(self, dtype, shape, source, depth=0, to_dtype=None, size_column=None, parent_array=None):
        self.DType = dtype
        self.Shape = shape
        self.Depth = depth
        self.ToDType = to_dtype
        self.Source = source
        self.SizeColumn = size_column
        self.ParentArray = parent_array
        
    @staticmethod
    def key(dataset, cname):
        return "%s:%s:@@desc.json" % (dataset, cname)

    def toDict(self):
        dct = dict(
                _Version = self.Version,
                type = self.DType,
                shape = self.Shape,
                depth = self.Depth,
                source = self.Source
        )
        if self.ToDType is not None and self.DType != self.ToDType:
            dct["type_np"] = self.self.ToDType
        if self.ParentArray:
            dct["parent_array"] = self.ParentArray
        if self.SizeColumn:
            dct["size_column"] = self.SizeColumn
        return dct
    
class ProvenanceSegment:

        def __init__(self, file_name, begin_event = 0, nevents = 0):
                self.FileName = file_name
                self.BeginEvent = begin_event
                self.NEvents = nevents

        def toDict(self):
                return dict(
                        FileName        =       self.FileName,
                        BeginEvent      =       self.BeginEvent,
                        NEvents         =       self.NEvents
                )

class RGInfo:

        Version = 3.0

        def __init__(self, rgid, segments, profile=None):
                self.RGID = rgid
                if isinstance(segments, ProvenanceSegment):
                    segments = [segments]
                self.Provenance = segments
                self.Profile = profile
                
        def addSegment(self, segment):
            self.Provenance.append(segment)
        
        @staticmethod
        def key(dataset, rgid):
            return rginfo_key(dataset, rgid)

        def toDict(self):
            nevents = sum([s.NEvents for s in self.Provenance])
            dct = dict(
                    _Version     = self.Version,
                    RGID         = self.RGID,
                    NEvents      = nevents,
                    Segments     = [s.toDict() for s in self.Provenance]
            )
            if self.Profile is not None:
                dct["Profile"] = self.Profile
            return dct
                
