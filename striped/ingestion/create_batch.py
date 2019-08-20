from striped.ingestion import Batch, FileInfo
from striped.client import CouchBaseBackend
from pythreader import TaskQueue, Task
from DataReader import DataReader

import time, sys, getopt, os, glob, stat, json


use_tqdm = False
try:
        from tqdm import tqdm
        use_tqdm = True
except:
        pass



class GetNEventsTask(Task):

    def __init__(self, file_info, schema, tick):
        Task.__init__(self)
        self.FileInfo = file_info
        self.Schema = schema
        self.Tick = tick
        
    def run(self):
        reader = DataReader(self.FileInfo.Path, self.Schema)
        self.FileInfo.NEvents = reader.nevents()
        if self.Tick is not None:
            self.Tick()
        

def create_batch(argv):
    from DataReader import DataReader

    Usage = """
    python create_batch.py [options] <batch_file> <bucket name> <dataset name> @<file with input file list>
    python create_batch.py [options] <batch_file> <bucket name> <dataset name> <directory path>
    python create_batch.py [options] <batch_file> <bucket name> <dataset name> <file> <file> ...

    Options:
        -O (reuse|REUSE|allocate) - override existing batch description file and either 
                              reuse same frame id range or 
                              allocate new range
        -c <couchbase config file>
        -n <target frame size>, default = 10000
        -p <path prefix> - prefix to add to the file paths read from the file or given as the list
        -k <n> - how many lowest path components, in addition to the file name 
                 to keep in the provenance data, defailt=0, i.e. keep the file name only
        -x <extension> - if the input is specified as a directory, then this is the extension of data files
                         under the directory. Default = "root"
        -m <n readers> - run multiple data reader threads in parallel, default = 1
        -q - be quiet
    """

    opts, args = getopt.getopt(argv, "n:p:k:x:O:qc:m:")
    opts = dict(opts)
    Config = opts.get("-c")
    FrameSize = int(opts.get("-n", 10000))
    Prefix = opts.get("-p")
    Keep = int(opts.get("-k", 0))
    Extension = opts.get("-x", "root")
    Override = "-O" in opts
    OverrideMode = opts.get("-O")
    MaxReaders = int(opts.get("-m", 1))

    Quiet = "-q" in opts

    if len(args) < 4 or not OverrideMode in (None, "reuse", "allocate", "REUSE"):
        print(Usage)
        sys.exit(1)

    BatchFile, BucketName, DatasetName = args[:3]

    exists = False
    try:
        exists = os.path.isfile(BatchFile)
    except:
            pass

    old_batch = None

    if exists:

        if not Override:
            print()
            print("ERROR: File %s exists. Use -O (reuse|allocate) to override." % (BatchFile,))
            print("Old file left unchanged.")
            print()
            print(Usage)
            sys.exit(1)

        old_batch = Batch.load(BatchFile)
        #print len(old_batch)

    source = args[3]
    if source[0] == '@':
        paths = [f 
            for f in [ff.strip() for ff in open(source[1:], "r").readlines()] 
            if f]
    elif stat.S_ISDIR(os.stat(source).st_mode):
        assert not Prefix, "\nERROR: Can not use path prefix with the input specified as the directory\n"
        if Extension[0] == '.':
            Extension = Extension[1:]
        paths = sorted(glob.glob("%s/*.%s" % (source, Extension)))      # local directory - assume root files
    else:
        paths = args[3:]            # explicit file path list

    if Prefix:  paths = [Prefix+f for f in paths]

    provenance_names = []
    for fp in paths:
        parts = fp.split("/")
        provenance_names.append("/".join(parts[-1-Keep:]))

    if not Quiet:
            print("Building frame map from %d files..." % (len(paths,)))

    backend = CouchBaseBackend(BucketName, print_errors = True, config = Config)
    schema = backend.schema(DatasetName)

    if not schema:
        print ("Dataset %s not found" % (DatasetName,))
        sys.exit(1)

    class FileCounter(object):
    
        def __init__(self, ntotal, show_tqdm):
            self.T = None if not (use_tqdm and show_tqdm) else tqdm(total=ntotal)
            
        def tick(self, n=1):
            if self.T is not None:
                self.T.update(n)
            

    file_counter = FileCounter(len(paths), not Quiet)
    file_infos = [FileInfo(path, None, prov) for path, prov in zip(paths, provenance_names)]
    queue = TaskQueue(MaxReaders, tasks = [GetNEventsTask(fi, schema, file_counter.tick) for fi in file_infos])
    queue.waitUntilEmpty()
            
    batch = Batch().build(DataReader, schema, FrameSize, file_infos)

    NFrames = len(batch)

    if not Quiet:
            print("Frame map with %d frames generated" % (NFrames,))
    start_farme_id = None
    if old_batch is not None:
        nold = len(old_batch)
        if OverrideMode.lower() == "reuse":
                if nold < NFrames and OverrideMode != "REUSE":
                    print()
                    print("ERROR: Can not reuse old frame id range because old range (%d) is shorter than needed (%d)" % (nold, NFrames))
                    print("       Use -O REUSE (capitals) to override")
                    print()
                    sys.exit(1)
                if nold > NFrames:
                    print()
                    print("WARNING: old frame id range (%d) is larger than new one (%d)" % (nold, NFrames))
                    print()
                start_farme_id = old_batch.StartFrameID
                if not Quiet:       print("Frame ID range starting at %d will be reused" % (start_farme_id,))

    if start_farme_id is None:
        start_farme_id = backend.allocateRGIDs(DatasetName, NFrames)
        if not Quiet:       print("Frame ID range is allocated starting at %d" % (start_farme_id,))

    batch.setStartFrameID(start_farme_id)

    batch.save(BatchFile)

    if not Quiet:   print("Batch saved to file: %s" % (BatchFile,))

if __name__ == '__main__':
    create_batch(sys.argv[1:])
