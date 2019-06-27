import ctypes
import glob
import os, time
import sys, getopt, json, pprint
from couchbase import FMT_BYTES, FMT_JSON
from couchbase.bucket import Bucket
from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError
import numpy as np
from striped.client import CouchBaseBackend, CouchBaseConfig
from striped.common import ColumnDescriptor


Usage = """
python createDataset.py [-c <config file>] <schema.json> <bucket name> <dataset>
"""

SchemaVersion = "3.0"

""" schema JSON file format

{
    "version":"2.2",
    "tree_top":"....",
    "attributes":
    {
        "path.to.attr": {
            dtype:"dtype",
            shape:...,
            source:...
            },
        ...
    },
    "branches":
    {
        "path.to.branch":
        {
            "relative.path.to.attr": {
                "source":"...",
                "dtype":"dtype"
            },
            ...
        }
    }
}
"""


convert_to = {
        "boolean":      "i1"
}       



config = None

opts, args = getopt.getopt(sys.argv[1:], "c:i")
for opt, val in opts:
    if opt == "-c": config = val

opts = dict(opts)
config = opts.get("-c")
reinit = "-i" in opts

if len(args) < 3:
	print Usage
	sys.exit(1)

schema_file, bucket_name, Dataset = args

schema = json.load(open(schema_file, "r"))
if not "version" in schema:
    schema["version"] = SchemaVersion

def parseSchema(schema):
    return schema["attributes"], schema["branches"]

fields, branches = parseSchema(schema)

backend = CouchBaseBackend(bucket_name, config=config)

key = "%s:@@schema.json" % (Dataset,)
backend[key].json = schema

for fn, fd in fields.items():
        ft = fd["dtype"]        
        fn = str(fn)
        ft = str(ft)
        shape = fd.get("shape", [])
        desc = ColumnDescriptor(
            ft, shape, fd["source"],
            size_column = fn + ".@size" if (shape and shape[0] is None) else None
        )
        key = ColumnDescriptor.key(Dataset, fn)
        backend[key].json = desc.toDict()
        #print key, desc

for branch, items in branches.items():
        for fn, fd in items.items():
                ft = fd["dtype"]        
                path = branch + "." + fn if fn else branch
                desc = ColumnDescriptor(
                    ft, fd.get("shape", []), fd["source"],
                    depth = 1,
                    parent_array = branch,
                    size_column = branch + ".@size"
                )
                
                key = ColumnDescriptor.key(Dataset, path)
                backend[key].json = desc.toDict()

next_rgid_name = "%s:@@nextRGID" % (Dataset,)
cb = backend.bucket
cb.remove(next_rgid_name, quiet=True)
value = backend.counter(next_rgid_name, initial=0).value
print "NextRGID counter created with value", value
