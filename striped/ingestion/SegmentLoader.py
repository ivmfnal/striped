import os, time, itertools
import sys, getopt, json, pprint
from couchbase import FMT_BYTES, FMT_JSON
from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError
import numpy as np
#from RowGroupMap import RowGroupMap
from striped.client import CouchBaseBackend
from striped.common import Stopwatch, Tracer
from striped.common import stripe_key, format_array, rginfo_key, RGInfo, ProvenanceSegment
from striped.pythreader import DEQueue, PyThread

from batch import SplitSegment, MergeSegment

#from trace import Tracer

SIZE_ARRAY_DTYPE = "<u4"
StripeHeaderFormatVersion = "1.0"

def stripeArray(groups, array):
    i = 0
    for n in groups:
        yield array[i:i+n]
        i += n


class StripedWriter(PyThread):

        BUFSIZE = 5
        QUEUESIZE = 50
        
        def __init__(self, backend, dataset_name):
            PyThread.__init__(self)
            self.Queue = DEQueue(self.QUEUESIZE)
            self.DatasetName = dataset_name            
            self.Backend = backend
            self.Buf = {}
            self.TotalBytes = 0
            self.Close = False
            self.T = Tracer()
            
        def add(self, rgid, column_name, array):
            self.Queue.append((rgid, column_name, array))
            
        def close(self):
            self.Close = True
            
        def run(self):
            while not (self.Close and len(self.Queue) == 0):
                rgid, column_name, array = self.Queue.pop()
                self.write(rgid, column_name, array)
            self.flush()

        def write(self, rgid, column_name, array):
            key = stripe_key(self.DatasetName, column_name, rgid)
            self.Buf[key] = format_array(array)
            if len(self.Buf) >= self.BUFSIZE:
                    self.flush()

        def flush(self):
            if self.Buf:
                with self.T["flush/put_data"]:
                    self.Backend.put_data(self.Buf)
                # verify
                if False:
                    with self.T["flush/verify"]:
                        read_back = self.Backend.get_data(self.Buf.keys())
                        for k, d in self.Buf.items():
                            rbd = read_back.get(k)
                            if rbd != d:
                                raise RuntimeError("Read-back data verification failed for key %s: data length:%d, read back:%d" % (
                                    k, len(d), len(rbd)))
                self.TotalBytes += sum([len(data) for data in self.Buf.values()])
            self.Buf = {}

class MergeLoader(object):

    def __init__(self, segment, frame_id, dataset_name, data_reader_class, schema, metadata, backend, 
                dry_run = False):
        
        self.Meta = metadata
        self.Files = segment.FilesSizes          # [(fp, pn, fs)...]
        self.Schema = schema
        self.FrameID = frame_id
        self.DryRun = dry_run
        self.DataReaderClass = data_reader_class
        self.DatasetName = dataset_name
        self.Backend = backend
        
    def run(self):
        writer = StripedWriter(self.Backend, self.DatasetName)
        writer.start()

        rgid = self.FrameID
        
        data_readers = [self.DataReaderClass(file_path, self.Schema) for file_path, _, _ in self.Files]
        provenance_names = [provenance_name for _, provenance_name, _ in self.Files]
        schema = self.Schema
        num_events = [data_reader.nevents() for data_reader in data_readers]
        meta_records = [data_reader.profile() for data_reader in data_readers]
        metadata = meta_records[0]
        assert all([mr == metadata for mr in meta_records]), "All files withing a frame must have the same metadata.\n" + \
                "  This frame id = %d, files: %s, metadata for first file: %s" % (self.FrameID, provenance_names, metadata)
        metadata = metadata or None
        dry_run = self.DryRun

        #
        # Delete RGInfo records
        #        
        if not dry_run:
            key = rginfo_key(self.DatasetName , rgid)
            del self.Backend[key]

        #
        # writing attributes
        #
        #print "schema:", schema
        for attr_name, attr_desc in schema["attributes"].items():
            data_stripes = []
            size_stripes = []
            
            for dr, n in zip(data_readers, num_events):
                data_stripe, size_stripe = next(dr.stripesAndSizes([n], None, attr_name, attr_desc))
                data_stripe = np.asarray(data_stripe, dtype=attr_desc["dtype"])
                if size_stripe is not None:
                    size_stripe = np.asarray(size_stripe, dtype=SIZE_ARRAY_DTYPE)
                data_stripes.append(data_stripe)
                size_stripes.append(size_stripe)
                
            if not dry_run and data_stripes:
                data_stripe = np.concatenate(data_stripes)
                writer.add(rgid, attr_name, data_stripe)
                if size_stripes[0] is not None:
                    size_stripe = np.concatenate(size_stripes)
                    writer.add(rgid, attr_name + ".@size", size_stripe)
                    

        #
        # writing branches
        #

        for bname, bdict in schema["branches"].items():
            branch_size_arrays = [reader.branchSizeArray(bname) for reader in data_readers]
            branch_size_array = np.asarray(np.concatenate(filter(lambda x: len(x) > 0, branch_size_arrays)), dtype=SIZE_ARRAY_DTYPE)

            writer.add(rgid, bname + ".@size", branch_size_array)

            for attr_name, attr_desc in bdict.items():
            
                data_stripes = []
                size_stripes = []
                
                for n, reader in zip(num_events, data_readers):
                    data, size = next(reader.stripesAndSizes([n], bname, attr_name, attr_desc))
                    if data is not None:
                        data_stripes.append(data)
                        if size is not None:
                            size_stripes.append(size)
                
                if not dry_run:       
                    if len(data_stripes):
                        data_stripe = np.asarray(np.concatenate(data_stripes), dtype=attr_desc["dtype"])
                        writer.add(rgid, bname+"."+attr_name, data_stripe)

                    if len(size_stripes):
                        size_stripe = np.asarray(np.concatenate(size_stripes), dtype=SIZE_ARRAY_DTYPE)
                        writer.add(rgid, bname+"."+attr_name+".@size", size_stripe)
                
        if not dry_run:
            writer.close()
            writer.join()

        #
        # Write RGInfo record
        #

        segments = [ProvenanceSegment(provenance_name, nevents = n)
                    for provenance_name, n in zip(provenance_names, num_events)]
        rginfo = RGInfo(rgid, segments, profile=metadata)
        if not dry_run:
            self.Backend.put_json({rginfo_key(self.DatasetName, rgid): rginfo.toDict()})


class SplitLoader(object):

    def __init__(self, segment, frames_to_load, dataset_name, data_reader_class, schema, metadata, backend, dry_run):
        
        assert isinstance(segment, SplitSegment)
        self.Segment = segment
        self.Meta = metadata
        self.FilePath = segment.Path
        self.ProvenanceName = segment.ProvenanceName
        self.Schema = schema
        self.FrameSizes = segment.FrameSizes
        self.StartFrameID = segment.StartFrameID
        self.DryRun = dry_run
        self.DataReaderClass = data_reader_class
        self.DatasetName = dataset_name
        self.Backend = backend
        self.FramesToLoad = frames_to_load
        
    def run(self):
        writer = StripedWriter(self.Backend, self.DatasetName)
        writer.start()
        
        schema = self.Schema
        groups = self.FrameSizes
        rgids_to_use = range(self.StartFrameID, self.StartFrameID+len(self.FrameSizes))
        dry_run = self.DryRun
        data_reader = self.DataReaderClass(self.FilePath, self.Schema)
        metadata = data_reader.profile() or None

        #
        # Delete RGInfo records
        #        
        if not dry_run:
                for r, _ in enumerate(groups):
                        rgid = rgids_to_use[r]
                        if rgid in self.FramesToLoad:
                            key = rginfo_key(self.DatasetName , rgid)
                            del self.Backend[key]

        #
        # writing attributes
        #

        for attr_name, attr_desc in schema["attributes"].items():
            for ig, (data_stripe, size_stripe) in \
                        enumerate(data_reader.stripesAndSizes(groups, None, attr_name, attr_desc)):
                rgid = rgids_to_use[ig]
                if rgid in self.FramesToLoad:
                    data_stripe = np.asarray(data_stripe, dtype=attr_desc["dtype"])
                    if not dry_run:
                        writer.add(rgid, attr_name, data_stripe)
                    if size_stripe is not None:
                        size_stripe = np.asarray(size_stripe, dtype=SIZE_ARRAY_DTYPE)
                        if not dry_run:
                            writer.add(rgid, attr_name + ".@size", size_stripe)

        #
        # writing branches
        #

        for bname, bdict in schema["branches"].items():
            branch_size_array = data_reader.branchSizeArray(bname)
            if branch_size_array is None or len(branch_size_array) == 0:    continue
            branch_size_array = np.asarray(branch_size_array, dtype=SIZE_ARRAY_DTYPE)

            for ig, size_stripe in enumerate(stripeArray(groups, branch_size_array)):
                rgid = rgids_to_use[ig]
                if rgid in self.FramesToLoad and not dry_run:
                    writer.add(rgid, bname + ".@size", size_stripe)

            for attr_name, attr_desc in bdict.items():
                for ig, (data_stripe, size_stripe) in enumerate(data_reader.stripesAndSizes(groups, bname, attr_name, attr_desc)):
                    rgid = rgids_to_use[ig]
                    if rgid in self.FramesToLoad:
                        if data_stripe is not None:
                            data_stripe = np.asarray(data_stripe, dtype=attr_desc["dtype"])
                            if not dry_run:
                                writer.add(rgid, bname+"."+attr_name, data_stripe)
                        if size_stripe is not None:
                            size_stripe = np.asarray(size_stripe, dtype=SIZE_ARRAY_DTYPE)
                            if not dry_run:
                                writer.add(rgid, bname+"."+attr_name+".@size", size_stripe)
                #data_reader.reopen()

        if not dry_run:
            writer.close()
            writer.join()

        #
        # Write RGInfo records
        #

        rginfos = {}

        ievent = 0

        for r, gs in enumerate(groups):
                rgid = rgids_to_use[r]
                if rgid in self.FramesToLoad:
                    key = rginfo_key(self.DatasetName , rgid)
                    segment = ProvenanceSegment(self.ProvenanceName, begin_event = ievent, nevents = gs)
                    rginfo = RGInfo(rgid, segment, profile=metadata)
                    rginfos[key] = rginfo.toDict()
                ievent += gs

        if not dry_run and rginfos:
            self.Backend.put_json(rginfos)

def segmentLoader(segment, frames_to_load, dataset_name, data_reader_class, schema, metadata, backend, dry_run = False):
    assert schema is not None
    if isinstance(segment, MergeSegment):
        assert len(frames_to_load) == 1
        frame_id = list(frames_to_load)[0]
        return MergeLoader(segment, frame_id, dataset_name, data_reader_class, schema, metadata, backend, dry_run)
    elif isinstance(segment, SplitSegment):
        return SplitLoader(segment, frames_to_load, dataset_name, data_reader_class, schema, metadata, backend, dry_run)
            
  
