#!/bin/env python 

import getopt, sys, os

def listDataset(argv):
    from couchbase.views.iterator import View
    from couchbase.views.params import Query
    from striped.client import CouchBaseBackend

    from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError

    Usage = """
    python listDataset.py -c <CouchBase config file> [-f|-l] <bucket name> <dataset name>
    """

    config_file = None

    opts, args = getopt.getopt(argv, "c:lfn")
    opts = dict(opts)
    config_file = opts.get("-c")
    files_only = "-f" in opts
    long_print = "-l" in opts
    counter = "-n" in opts

    if len(sys.argv) < 2:
        print(Usage)
        sys.exit(1)

    bucket_name, dataset_name = args
    backend = CouchBaseBackend(bucket_name, config=config_file)
    bucket = backend.bucket

    if False:
            q = Query()
            q.mapkey_single = dataset_name
            v = View(bucket, "views", "RGInfos", query=q)
            infos = [x.value for x in v if x.key == dataset_name]
    infos = backend.RGInfos(dataset_name)
    infos = sorted(infos, key = lambda info: info["RGID"])


    if long_print:
            print("RGID    NEvents    File(s)")
            print("------- ---------- -------")

            nevents = 0

            files = {}
            rgids = set()


            for info in infos:
                fn = info["Segments"][0]["FileName"]
                print("%7d %10d %s" % (info["RGID"], info["NEvents"], fn))
                rgids.add(info["RGID"])
                files[fn] = 1
                for s in info["Segments"][1:]:
                    print("%19s %s" % (" ", s["FileName"]))
                    files[s["FileName"]] = 1
                nevents += info["NEvents"]

            print("------- ---------- -------")
            print("%7d %10d %d" % (len(infos), nevents, len(files)))   

            maxrgid = max(rgids)
            if len(rgids) != maxrgid+1:
                    print("Missing RGIDs (%d):" % (maxrgid+1 - len(rgids),))
                    for rgid in range(maxrgid):
                            if not rgid in rgids:
                                    print(rgid, end=' ')
                    print()
    elif files_only:
            files = {}          # filename -> nevents
            for info in infos:
                for s in info["Segments"]:
                    fn = s["FileName"]
                    files[fn] = files.get(fn, 0) + s["NEvents"]
            for fn in sorted(files.keys()):
                    print(fn)

    else:
            files = set()
            rgids = set()
            nevents = 0

            try:        
                counter =  backend.counter("%s:@@nextRGID" % (dataset_name,), delta=0).value
            except NotFoundError:
                    counter = None

            for info in infos:
                rgids.add(info["RGID"])
                for s in info["Segments"]:
                    files.add(s["FileName"])
                nevents += info["NEvents"]
            print("Next FrameID:      ", counter)
            print("Files:             ", len(files))
            print("Frames:            ", len(rgids))
            print("Events:            ", nevents)
            if len(rgids):
                print("Max farme id:      ", max(rgids))
                print("Events/frame:      ", int(float(nevents)/float(len(rgids))+0.5))

                maxrgid = max(rgids)
                if len(rgids) < maxrgid + 1:
                        print("Missing RGIDs (%d):" % (maxrgid+1 - len(rgids),))
                        for rgid in range(maxrgid):
                                if not rgid in rgids:
                                        print(rgid, end=' ')
                        print()
                
        
if __name__ == "__main__":
    listDataset(sys.argv[1:])

