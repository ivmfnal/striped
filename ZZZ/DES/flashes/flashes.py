from striped.job import Session
from striped.common import Meta
from striped.job.callbacks import ProgressBarCallback
import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import sys, time, getopt
from pythreader import Task, TaskQueue, Primitive, synchronized
import fitsio

from striped.common import Tracer
T = Tracer()

Usage = """
python all_hpix.py
"""

opts, args = getopt.getopt(sys.argv[1:], "?h")
opts = dict(opts)

if "-?" in opts or "-h" in opts:
    print Usage
    sys.exit(1)


Min_Objects = 10
Mean_Cut = 5.0


#job_server_address = ("dbwebdev.fnal.gov", 8765) 

class CountsCollector:

    def __init__(self, session):
        self.Counts = []       # hpix -> (nobjects, nobservations)
        self.MinHPIX = None
        self.MaxHPIX = None
        self.Session = session
        
    def on_job_start(self, job):
        print "CountsCollector: job %s started. %d total objects, %d selected in %d frames" % (
            job.JID, job.EventsInDataset, job.EventsToProcess, len(job.FramesToProcess))

    def on_streams_update(self, nevents, data):
        if "message" in data:
            for msg in data["message"]:
                print msg
        if "hpix_stats" in data:
            for hpix, obj_count, obs_count in data["hpix_stats"]:
                self.Counts.append((hpix, obj_count, obs_count))
            
    def run(self):
        job = self.Session.createJob("Bliss", 
                            user_params = {"min_objects":Min_Objects},
		                    use_data_cache = True, 
                            callbacks = [self, ProgressBarCallback()],
                            worker_class_file="count_observations_worker.py")
        job.run()
        
        counts = {}
        
        for i, (hpix_arr, nobj_arr, nobs_arr) in enumerate(self.Counts):
            for j, hpix in enumerate(hpix_arr):
                nobj, nobs = counts.get(hpix, (0,0))
                counts[hpix] = (nobj+nobj_arr[j], nobs+nobs_arr[j])
        #for hpix, (nobj, nobs) in sorted(counts.items()):
        #    if nobs > nobj:
        #        print hpix, nobj, nobs
        counts_array = np.empty((len(counts),3), dtype=np.int64)
        for i, (hpix, (nobj, nobs)) in enumerate(sorted(counts.items())):
            counts_array[i,0] = hpix
            counts_array[i,1] = nobj
            counts_array[i,2] = nobs
            
        return counts_array

class FlashFinder(Task):

    def __init__(self, session, pixels):
        Task.__init__(self)
        self.Session = session
        self.FlashLists = []
        self.Flashes = []
        self.FlashesFound = 0
        self.FlashOID = []
        self.FlashHPIX = []
        self.Pixels = pixels
        self.MinHPIX, self.MaxHPIX = pixels[0], pixels[-1]
                
    def on_job_start(self, job):
        self.JID = job.JID
        print "FlashFinder: job %s started. %d total objects, %d selected in %d frames" % (
            job.JID, job.EventsInDataset, job.EventsToProcess, len(job.FramesToProcess))

    def on_streams_update(self, nevents, data):
        if "flashes" in data:
            flashes = data["flashes"]
            self.FlashLists += flashes
            for fl in flashes:
                if len(fl):
                    self.FlashesFound += len(fl)
        
                
    def on_job_finish(self, nobjects, error):
        if error:
            print "FlashFinder: job %s failed: %s" % (self.JID, error)
        else:
            print "FlashFinder: job %s ended. %d objects processed. %d flashes found" % (self.JID, nobjects, 
                    self.FlashesFound)
    
                
            
    def run(self):
        #print "Finder %d:%d started" % (self.Pixels[0], self.Pixels[-1])
        job = self.Session.createJob("Bliss", 
		                    use_data_cache = True, 
                            callbacks = self,
                            user_params = dict(pixels=self.Pixels),
                            frame_selector = (Meta("hpix_min") <= self.MaxHPIX) & (Meta("hpix_max") >= self.MinHPIX),
                            #("and", ("le", "hpix_min", self.MaxHPIX), ("ge", "hpix_max", self.MinHPIX)),
                            worker_class_file="find_flashes_worker.py")
        job.run()
        
        if len(self.FlashLists):
            self.Flashes = np.concatenate(self.FlashLists)
            
                
session = Session("striped.yaml")

print "Calculating pixel statistics..."

counter = CountsCollector(session)
counts = counter.run()

print "pixels with >= %d objects: %d" % (Min_Objects, len(counts))

nobj = counts[:,1]
nobs = counts[:,2]

mean_count = np.asarray(nobs, dtype=np.float64)/nobj        # missing pixels will have mean_count=0 anyway


counts = counts[mean_count > Mean_Cut]
mean_count = mean_count[mean_count > Mean_Cut]


print "pixels with mean_count > %s: %d" % (Mean_Cut, len(counts))

pixels = counts[:,0]
npixels = len(pixels)

print "npixels:", npixels

hpix_min = pixels[0]
hpix_max = pixels[-1]

mean_count_map = np.empty((hpix_max-hpix_min+1,), dtype=mean_count.dtype)
mean_count_map[pixels-hpix_min] = mean_count

njobs = min(15, len(pixels))
pixels_per_job = (npixels+njobs-1)/njobs
finders = [FlashFinder(session, pixels[i:i+pixels_per_job])
        for i in range(0, npixels, pixels_per_job)]
#finders = finders[:1]
q = TaskQueue(10, tasks=finders)
q.waitUntilEmpty()
ntotal = 0

all_flashes = []

for f in finders:
    out = f.Flashes
    #print type(out), len(out)
    all_flashes.append(out)

#print all_flashes
    
if len(all_flashes):
    all_flashes = np.concatenate(all_flashes)
    fitsio.write("flashes.fits", all_flashes, clobber=True)
    print "%d found flashes stored in flashes.fits" % (len(all_flashes),)



