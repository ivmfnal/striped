import os, time, itertools
import sys, getopt, json, pprint
from couchbase import FMT_BYTES, FMT_JSON
from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError
import numpy as np
from numpy.lib.recfunctions import append_fields
import uproot
from striped.client import CouchBaseBackend
import fitsio, healpy

Usage = """
python ingest.py <schema.json> <bucket name> <dataset name> <file> [<file>...]
"""

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

        Version = "3.0"

        def __init__(self, rgid, segment):
                self.RGID = rgid
                self.NEvents = segment.NEvents
                self.Provenance = [segment]

        def toDict(self):
                return dict(
                        _Version     = self.Version,
                        RGID         = self.RGID,
                        NEvents      = self.NEvents,
                        Segments     = [s.toDict() for s in self.Provenance]
                )



Fields = ["COADD_OBJECT_ID","ALPHAWIN_J2000","DELTAWIN_J2000"]
StripeHeaderFormatVersion = "1.0"

def ingest_file(backend, schema, filename, dataset):
    data = fitsio.read(filename, columns=Fields)
    data = append_fields(data, "HPIX", healpy.ang2pix(16384, 
                theta=data["ALPHAWIN_J2000"],
                phi=data["DELTAWIN_J2000"],
                lonlat=True, nest=True)
            )
    data.sort(order="HPIX")
    
    rgid = backend.counter("%s:@@nextRGID" % (dataset,), delta=1).value-1
    arrays = {}
    for field_name, desc in schema["attributes"].items():
        source = desc["source"]
        key = "%s:%s:%d.bin" % (dataset, field_name, rgid)
        if not source and field_name == "HPIX":
            source = "HPIX"
        if source:
            arr = np.asarray(data[source], dtype=desc["dtype"]).copy()
            header = "#__header:version=%s;dtype=%s#" % (StripeHeaderFormatVersion, arr.dtype.str)
            arrays[key] = bytes(header) + bytes(arr.data) 
    backend.put_data(arrays)
    
    rginfo = RGInfo(rgid, ProvenanceSegment(filename.rsplit("/",1)[-1], 0, len(data)))
    key = "%s:@@rginfo:%s.json" % (dataset, rgid)
    backend[key].json = rginfo.toDict()
    
    print "File %s ingested with %d objects, hpix range: %d %d" % (filename, len(data), data["HPIX"][0], data["HPIX"][-1])
            
        

opts, args = getopt.getopt(sys.argv[1:], "")

if not args:
    print Usage
    sys.exit(1)

schema = json.load(open(args[0], "r"))
bucket_name = args[1]
dataset = args[2]
files = args[3:]
config_file = None

backend = CouchBaseBackend(bucket_name, print_errors = True, config = config_file)

for filename in files:
    ingest_file(backend, schema, filename, dataset)
