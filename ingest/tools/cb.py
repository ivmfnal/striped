import json, getopt, sys
import numpy as np
from striped.client import CouchBaseBackend 

Usage = """
python cb.py get [-j] [-o <file>|-d <dtype>] <bucket> <key>
	-n means show data as numpy array of given dtype and shape

python cb.py put [-j] [-f <file>|-d <data>] <bucket> <key> 
"""

if not sys.argv[1:]:
    print Usage
    sys.exit(1)
    
cmd = sys.argv[1]
args = sys.argv[2:]
if cmd == "get":

    show_as_np = False

    opts, args = getopt.getopt(args, "d:jo:")
    opts = dict(opts)
    dtype = opts.get("-d")
    out_file = opts.get("-o")
    json_data = "-j" in opts

    Bucket, Key = args

    cb = CouchBaseBackend(Bucket)
    if json_data:
        data = cb[Key].json
        out = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '))
        out_file = open(out_file, "w") if out_file else sys.stdout
        out_file.write(out)            
    else:
        data = cb[Key].data
        if out_file:
            open(out_file, "wb").write(data)
        elif dtype:
            data = np.frombuffer(data, dtype=dtype)
            print data.shape, data.dtype, data
        else:
            print len(data), repr(data[:100])

elif cmd == "put":

    opts, args = getopt.getopt(args, "jf:d:")
    opts = dict(opts)
    Bucket, Key = args
    json_in = "-j" in opts
    data = None
    if "-d" in opts:
        data = opts["-d"]
    else:
        data = open(opts["-f"], "rb").read()
    if json_in:
        data = json.loads(data)
    
    cb = CouchBaseBackend(Bucket)
    if json_in:
        cb[Key].json = data
    else:
        cb[Key].data = data
