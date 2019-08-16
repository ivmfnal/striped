#!/bin/env python 

import sys, getopt, glob


def verifyDataset(argv):
    from striped.client import CouchBaseBackend
    import uproot

    Usage = """
    python validateDaatset.py [-b] <dataset directory> <root tree top> <bucket>

     -b means check binary records only

    """

    opts, args = getopt.getopt(argv, "b")
    opts = dict(opts)
    binary_only = "-b" in opts
    dataset_dir, tree_top, bucket_name = args

    dataset_name = dataset_dir.split("/")[-1]
    backend = CouchBaseBackend(bucket_name)

    rgids = list(backend.RGIDs(dataset_name))
    print("I: %d row groups in the dataset" % (len(rgids),))

    if not binary_only:

            files = {}      # filename -> events

            for f in glob.glob("%s/*.root" % (dataset_dir,)):
                fn = f.split("/")[-1]
                tree = uproot.open(f)[tree_top]
                files[fn] = tree.numentries

            print("I: %d files, %d events" % (len(files), sum(files.values())))


            files_in_dataset = {}
            rgids = set()
            total_events = 0

            for info in backend.RGInfos(dataset_name):
                rgid = info["RGID"]
                rgids.add(rgid)
                nevents = info["NEvents"]
                total_events += nevents
                nevents_in_segments = 0
                for s in info["Segments"]:
                    fn = s["FileName"]
                    ne = s["NEvents"]
                    files_in_dataset[fn] = files_in_dataset.get(fn, 0) + ne
                    nevents_in_segments += ne
                if nevents != nevents_in_segments:
                    print("E: Total number of events in RG #%d (%d) is not equal sum of events in segments (%d)" % (rgid, nevents, nevents_in_segments))

            if len(rgids) != max(rgids)+1:
                maxrgid = max(rgids)
                missing = [i for i in range(maxrgid+1) if not i in rgids]
                print("W: gap(s) in rgids. Missing %d rgids: %s" % (max(rgids)+1-len(rgids), missing))

            for f, n in files.items():
                if not f in  files_in_dataset:
                    print("E: File %s is not in the database" % (f,))
                else:
                    n_file = files[f]
                    n_db = files_in_dataset[f]
                    if n_file != n_db:
                        print("E: Number of events in file %s (%d) differs from the database (%d)" % (f, n_file, n_db))

    print("I: Scanning keys...")

    rgids_per_column = {}
    data_keys = set()

    nkeys = 0

    for k in backend.keys(dataset_name):
        # parse key
        parts = k.split(":")
        if len(parts) == 3:
            _, column, tail = parts
            if not tail.startswith('@'):
                tail_parts = tail.split(".")
                rgid = int(tail_parts[0])
                key_type = tail_parts[1]

                if rgid in rgids and key_type == 'bin':
                    data_keys.add("%s:%d" % (column, rgid))
                    column_rgids = rgids_per_column.get(column)
                    if not column_rgids:
                        column_rgids = set()
                        rgids_per_column[column] = column_rgids
                    column_rgids.add(rgid)

    print("I: %d data keys found for %d columns" % (len(data_keys), len(rgids_per_column.keys())))

    N = max([len(r) for r in rgids_per_column.values()])

    print("I: max %d data keys per column" % (N,))

    nmissing = 0

    for cn, r in rgids_per_column.items():
        n = len(r)
        if n != N:      
            print("E: %d data stripes are mising for column %s" % (N-n, cn))
            nmissing += N-n

    if nmissing:
            print("E: %d data stripes are missing" % (nmissing,))


if __name__ == "__main__":
    verifyDataset(sys.argv[1:])
    
