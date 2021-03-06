import json, sys

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

use_tqdm = False
try:
        import tqdm
        use_tqdm = True
except:
        pass


def distribute_items(N, n):
    k = N % n
    m = (N-k)//n
    i = 0
    out = [m+1]*k + [m]*(n-k)
    return out

def split_to_parts(total_size, target_part_size):
        if total_size < (4*target_part_size)//3:
                return [total_size]
        nparts = total_size//target_part_size
        part_size_0 = float(total_size)/nparts
        part_size_1 = float(total_size)/(nparts+1)
        if abs(part_size_0-target_part_size) > abs(part_size_1-target_part_size):
                nparts += 1
        return distribute_items(total_size, nparts)
        
class FileInfo(object):
    def __init__(self, path, nevents, provenance_name):
        self.Path = path
        self.NEvents = nevents
        self.ProvenanceName = provenance_name

class SplitSegment(object):

    def __init__(self, file_path, pn, frame_sizes):
        self.Path = file_path
        self.ProvenanceName = pn
        self.FrameSizes = frame_sizes
        self.StartFrameID = None            # used by the loader
        
    def __len__(self):
        return len(self.FrameSizes)
        
    def frameIDs(self):
        return list(range(self.StartFrameID, self.StartFrameID+len(self)))
        
    def toDict(self):
        return dict(
            type    =   "split",
            path    =   self.Path,
            file    =   self.ProvenanceName,
            frames  =   self.FrameSizes
        )

    @staticmethod
    def fromDict(dct):
        assert dct["type"] == "split"
        #print dct
        return SplitSegment(dct["path"], dct["file"], dct["frames"])
        
        
class MergeSegment(object):

    def __init__(self, files_sizes):
        self.FilesSizes = files_sizes
        self.StartFrameID = None            # used by the loader
        
    def __len__(self):
        return 1
        
    def frameIDs(self):
        return [self.StartFrameID]
        
    def toDict(self):
        return dict(
            type    =   "merge",
            parts    =   [
                {"path":fp, "file":pn, "size":fs} for fp, pn, fs in self.FilesSizes
            ]
        )
        
    @staticmethod
    def fromDict(dct):
        assert dct["type"] == "merge"
        return MergeSegment([(part["path"], part["file"], part["size"]) for part in dct["parts"]])
        
class FrameMap(object):

    def __init__(self, map=None):
        self.Map = map

    def assignFrameIDs(self, start_frame_id):
        fid = start_frame_id
        for s in self.Map:
            s.StartFrameID = fid
            fid += len(s)
        
    def __len__(self):
        # total number of frames
        return sum([len(x) for x in self.Map], 0)        
        
    def __iter__(self):
        return (s for s in self.Map)
        
    def __getitem__(self, i):
        return self.Map[i]

    @staticmethod
    def build(data_reader_class, schema, target_frame_size, file_infos):

        path_list = [f.Path for f in file_infos]
        file_sizes = [f.NEvents for f in file_infos]
        provenance_names = [f.ProvenanceName for f in file_infos]
        provenance_map = dict(list(zip(path_list, provenance_names)))
        segments = []
        current_segment = []
        current_segment_size = 0
        for fi in file_infos:
            fp, fs = fi.Path, fi.NEvents
            d_add = abs(current_segment_size + fs - target_frame_size)
            d_close = abs(current_segment_size - target_frame_size)
            
            if current_segment and d_close < d_add:
                # close current segment
                segments.append((current_segment, current_segment_size))
                current_segment = []
                current_segment_size = 0
            else:
                # add to current segment
                current_segment.append((fp, provenance_map[fp], fs))
                current_segment_size += fs
            
            if current_segment_size >= target_frame_size:
                segments.append((current_segment, current_segment_size))
                current_segment = []
                current_segment_size = 0

        if current_segment:
                segments.append((current_segment, current_segment_size))

        #pprint.pprint(segments)
        
        map = []

        # split files if necessary
        new_segments = []
        for segment, segment_size in segments:
            if len(segment) == 1:
                fp, fn, fs = segment[0]
                sizes = split_to_parts(segment_size, target_frame_size)
                map.append(SplitSegment(fp, fn, sizes))
            else:
                map.append(MergeSegment(segment))
        
        return FrameMap(map)
    
    @staticmethod        
    def fromJSON(json_or_object):
        if isinstance(json_or_object, str):
            json_or_object = json.loads(json_or_object)
        m = []
        for segment in json_or_object:
            #print "segment:", segment
            if segment["type"] == "split":
                m.append(SplitSegment.fromDict(segment))
            elif segment["type"] == "merge":
                m.append(MergeSegment.fromDict(segment))
        return FrameMap(m)
        
        
    def jsonable(self):
        return [s.toDict() for s in (self.Map or [])]


class Batch(object):

    def __init__(self, start_frame_id = None, frame_map = None):
        self.StartFrameID = start_frame_id
        self.FrameMap = frame_map
        if start_frame_id is not None:
            self.FrameMap.assignFrameIDs(start_frame_id)

    def setStartFrameID(self, fid):
        self.StartFrameID = fid
        self.FrameMap.assignFrameIDs(fid)
    
    def __len__(self):
        return len(self.FrameMap)
        
    def __iter__(self):
        return self.FrameMap.__iter__()
        
    def __getitem__(self, i):
        return self.FrameMap[i]
        
    def jsonable(self):
        return dict(
            n_frames = len(self.FrameMap),
            start_frame_id = self.StartFrameID,
            frame_map = self.FrameMap.jsonable()
        )
        
    @staticmethod 
    def build(data_reader_class, schema, frame_size, file_infos):
        return Batch(frame_map = FrameMap.build(data_reader_class, schema, frame_size, file_infos))
    
    @staticmethod
    def fromJSON(json_or_object):
        if isinstance(json_or_object, str):
            json_or_object = json.loads(json_or_object)
        return Batch(start_frame_id = json_or_object["start_frame_id"], frame_map = FrameMap.fromJSON(json_or_object["frame_map"]))
        
    @staticmethod
    def load(path_or_file):
        def json_object_hook(arg): # this hook will convert every unicode key or value to utf-8 while loading json

                def to_str(x):
                    if isinstance(x, list):
                        return [to_str(xx) for xx in x]
                    if PY3 and isinstance(x, bytes): return x.decode("utf-8", "ignore")
                    if PY2 and isinstance(x, (bytes, unicode)): return str(x)
                    return x

                return { to_str(k) : to_str(v) for k, v in arg.items() }

        if isinstance(path_or_file, str):
            path_or_file = open(path_or_file, "r")
        return Batch.fromJSON(json.load(path_or_file, object_hook=json_object_hook))
        
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.jsonable(), f, indent=4, sort_keys=True)

        
