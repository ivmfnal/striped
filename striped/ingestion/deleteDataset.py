#!/bin/env python 

def deleteDataset(argv):
    from couchbase.bucket import Bucket
    from couchbase.views.iterator import View
    from couchbase.views.params import Query
    from couchbase.exceptions import NotFoundError
    from striped.client import CouchBaseBackend, CouchBaseConfig
    import os, sys, json, getopt, random

    Usage="""
    python delete_dataset.py [-c <Couchbase config file>] <bucket> <dataset> 
    """

    def delete_metadata(backend, dataset):
        del backend["%s:@@nextRGID" % (dataset,)]
        keys = (k for k in backend.keys(dataset) if k.endswith(".json"))
        return backend.delete(keys)


    def delete_dataset(dataset, bucket, ratio):
        q = Query()
        q.mapkey_single = dataset
        v = View(bucket, "views", "keys", query=q)
        keys = (x.value for x in v if x.key == dataset)

        def pack_in_groups(keys, n, ratio):
            group = []
            for k in keys:
                if ratio > random.random():
                        #print k
                        if len(group) >= n:
                            #print(group[0])
                            yield group
                            group = []
                        group.append(k)
            if len(group) >= 0:
                yield group

        nremoved = 0

        for kg in pack_in_groups(keys, 500, ratio):
            try:
                if kg:      
                    bucket.remove_multi(kg, quiet=True)
            except NotFoundError as error:
                    print(error)
            else:
                nremoved += len(kg)
            if nremoved and nremoved % 10000 == 0:
                    print(nremoved)

        return nremoved

    config = None
    ratio = 1.0

    opts, args = getopt.getopt(argv, "c:r:m")
    opts = dict(opts)
    if "-c" in opts:        
            config = opts["-c"]

    ratio = float(opts.get("-r", 1.0))
    meta_only = "-m" in opts

    if not args:
            print(Usage)
            sys.exit(1)

    bucket_name = args[0]
    dataset_name = args[1]

    backend = CouchBaseBackend(bucket_name, config=config)
    bucket = backend.bucket
    n_meta = delete_metadata(backend, dataset_name)
    n_data = 0
    if not meta_only:
            n_data = delete_dataset(dataset_name, bucket, ratio)
    print(n_meta, "metadata items removed")
    print(n_data, "data items removed")


if __name__ == "__main__":
    import sys
    deleteDataset(sys.argv[1:])

