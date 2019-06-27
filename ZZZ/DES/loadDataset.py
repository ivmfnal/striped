import subprocess, time, sys, getopt, os, glob
from pythreader import TaskQueue, Subprocess, Task

class FileLoaderTask(Task):

    def __init__(self, inx, schema_file, files, bucket_name, dataset_name):
        Task.__init__(self)
        self.Inx = inx
        self.Files = files
        self.BucketName = bucket_name
        self.DatasetName = dataset_name
        self.SchemaFile = schema_file
        
    def run(self):
        command = ["python", "ingest.py", self.SchemaFile, self.BucketName, self.DatasetName] + self.Files
        print "\nStaring %d: %s" % (self.Inx, command,)
        sp = Subprocess(command, env=os.environ)
        sp.wait()
        print "\n%d is done" % (self.Inx,)


Usage = """
python loadDataset.py [options] <schema.json> <directory> <bucket name> <dataset name>

Options:
    -m <max workers>, default = 5
    -n <nfiles in batch>, default = 10
    -s <stagger>, default = 10 (seconds)
"""

MaxWorkers = 5
BatchSize = 10
Stagger = 10

opts, args = getopt.getopt(sys.argv[1:], "m:n:s:")
for opt, val in opts:
    if opt == "-m":     MaxWorkers = int(val)
    elif opt == "-n":   BatchSize = int(val)
    elif opt == "-s":   Stagger = int(val)

if len(args) != 4:
    print Usage
    sys.exit(1)
    
SchemaFile, Directory, BucketName, DatasetName = args

files = sorted(glob.glob("%s/*.fits" % (Directory,)))

tq = TaskQueue(MaxWorkers)
i = 0
while files:
    batch = files[:BatchSize]
    files = files[BatchSize:]
    t = FileLoaderTask(i, SchemaFile, batch, BucketName, DatasetName)
    tq << t
    time.sleep(Stagger)
    i += 1

    
