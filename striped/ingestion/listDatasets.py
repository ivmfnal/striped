import getopt, sys, os

def listDatasets(argv):
    from couchbase.views.iterator import View
    from striped.client import CouchBaseBackend

    from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError

    Usage = """
    python listDatasets.py -c <CouchBase config file> [-l] <bucket name> [<dataset name pattern>]
    """

    config_file = None

    opts, args = getopt.getopt(argv, "c:l")
    opts = dict(opts)
    config_file = opts.get("-c")
    long_print = "-l" in opts

    if len(args) < 1:
        print(Usage)
        sys.exit(1)

    bucket_name = args[0]
    pattern = None if len(args) < 2 else args[1]
    backend = CouchBaseBackend(bucket_name, config=config_file)
    bucket = backend.bucket
    
    if long_print:
        print("%-30s %6s %6s %15s" % ("Dataset", "Frames","Files","Events"))
        print("%-30s %6s %6s %15s" % ("-------", "------","-----","------"))

    for ds in sorted(backend.datasets()):
	    if long_print:
		    nevents = 0
		    nrgs = 0
		    files = set()
		    for rginfo in backend.RGInfos(ds):
			    nevents += rginfo["NEvents"]
			    nrgs += 1
			    for s in rginfo["Segments"]:
				    files.add(s["FileName"])
		    print("%-30s %6d %6d %15d" % (ds, nrgs, len(files), nevents))
	    else:
		    print(ds)





