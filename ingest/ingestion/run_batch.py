import subprocess, time, sys, getopt, os, glob, stat, multiprocessing, json, traceback
from striped.pythreader import TaskQueue, Subprocess, Task
from striped.ingestion import segmentLoader, Batch

from DataReader import DataReader

from striped.client import CouchBaseBackend

class LoaderSubprocess(multiprocessing.Process):

    def __init__(self, segment, config, schema, metadata, frames_to_do, 
                bucket_name, dataset_name, data_reader_class, dry_run):
        multiprocessing.Process.__init__(self)
	assert schema is not None
        self.Schema = schema
        self.Segment = segment
        self.FramesToDo = frames_to_do
        self.BucketName = bucket_name
        self.Config = config
        self.DatasetName = dataset_name
        self.DataReaderClass = data_reader_class
        self.Metadata = metadata
        self.DryRun = dry_run
        #print "LoaderSubprocess object cteated with frames_to_do=", self.FramesToDo
        
    def run(self):
        backend = CouchBaseBackend(self.BucketName, print_errors = True, config = self.Config)
        frames_to_do = self.FramesToDo
        segment_frames = set(self.Segment.frameIDs())
        print "Process %d started to load frames: %s" % (os.getpid(), sorted(list(segment_frames & frames_to_do)))
        if frames_to_do:
	    assert self.Schema is not None
            sl = segmentLoader(self.Segment, frames_to_do, self.DatasetName, self.DataReaderClass, 
                self.Schema, self.Metadata, backend, self.DryRun)
            try:
                sl.run()
            except:
                print "Process %d exiting with error:" % (os.getpid(),)
                traceback.print_exc()
                sys.exit(1)
            else:
                print "Process %d finished successfully" % (os.getpid(),)
            
class LoaderTask(Task):

    def __init__(self, controller, id, segment, *params):
        Task.__init__(self)
        self.ID = id
        self.Controller = controller
        self.Segment = segment
        self.Process = LoaderSubprocess(segment, *params)
        
    def run(self):
        try:    
            self.Process.start()
            self.Process.join()
        finally:
            self.Controller.processExited(self.ID, self.Process.exitcode)
            self.Controller = None
            
        

class Controller(object):

    def __init__(self, batch, max_workers, config, schema, bucket_name, dataset_name, data_reader_class,
                    existing_frames, dry_run):
        self.Batch = batch
        self.TaskQueue = TaskQueue(max_workers)
        self.ExistingFrames = existing_frames
        self.CBConfig = config
        self.Schema = schema
        self.BucketName = bucket_name
        sekf.DatasetName = dtaset_name
        self.DataReaderClass = data_reader_class
        self.DryRun = dry_run
            
    def run(self):
        for i, segment in enumerate(self.Batch):
            frames = set(segment.frameIDs()) - self.ExistingFrames
            if frames:
                task = LoaderTask(self, i, segment, self.CBConfig, self.Schema, None,       # FIX ME: metadata is None for now
                    frames, self.BucketName, self.DatasetName, 
                    self.DataReaderClass, self.DryRun)
                self.TaskQueue.add(task)
        self.TaskQueue.waitUntilDone()

class Printer:

    def __init__(self, batch):
        self.Batch = batch
        
    def processExited(self, i, exitcode):
        if exitcode:
            print "Loader for segment %s failed: %s" % (self.Batch[i], exitcode)
        else:
            #print "Segment %s loaded successfully" % (self.Batch[i],)
            pass
        
        

Usage = """
python run_batch.py [options] <batch file> <bucket name> <dataset name>

Options:
    -c <CouchBase config file>, default - value of the COUCHBASE_BACKEND_CFG environment variable
    -m <max workers>, default = 5
    -O - override existing frames
    -s <stagger>, default = 10 (seconds)
    -n - dry run
"""

opts, args = getopt.getopt(sys.argv[1:], "m:s:c:On")
opts = dict(opts)
MaxWorkers = int(opts.get("-m", 5))
Stagger = float(opts.get("-s", 1))
Config = opts.get("-c", os.environ.get("COUCHBASE_BACKEND_CFG"))
Override = "-O" in opts
DryRun = "-n" in opts

if not Config:
    print "Couchbase config file must be specified either with -c, or using COUCHBASE_BACKEND_CFG env. variable"
    print
    print Usage
    sys.exit(1)

if len(args) < 3:
    print Usage
    sys.exit(2)

batch_file, bucket_name, dataset_name = args

batch = Batch.load(batch_file)
backend = CouchBaseBackend(bucket_name, print_errors = True, config = Config)
schema = backend.schema(dataset_name)

if not schema:
	print "Empty schema"
	sys.exit(1)

existing_frames = set()

if not Override:
    existing_frames = set(backend.RGIDs(dataset_name))
    
if existing_frames:
    print "The following frames exist and will not be overriden:", sorted(list(existing_frames))

task_queue = TaskQueue(MaxWorkers, stagger=Stagger)
printer = Printer(batch)
for i, segment in enumerate(batch):
    #print "segment:", i, segment, segment.frameIDs()
    frames = set(segment.frameIDs()) - existing_frames
    #print "segment:", i, segment, segment.frameIDs(), frames
    if frames:
        task = LoaderTask(printer, i, segment, Config, schema, None,       # FIX ME: metadata is None for now
            frames, bucket_name, dataset_name, 
            DataReader, DryRun)
        task_queue.addTask(task)
task_queue.waitUntilEmpty()


