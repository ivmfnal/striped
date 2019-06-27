from striped.job import SinglePointStripedSession as Session
import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import fitsio, healpy as hp
import sys, time, getopt
from astropy.io.fits import Header

from pythreader import Task, TaskQueue, Primitive, synchronized

from striped.common import Tracer
T = Tracer()

def read_bliss_file(path):
    fits = fitsio.FITS(path)

    # The header for the image that this catalog was derived from is
    # stored in the first extension of the FITS file. However, it's format
    # is weird, so we parse it.

    # First we build a string
    hdrstr = '\n'.join(fits['LDAC_IMHEAD'].read()[0][0])
    # Then we use astropy to parse that string into a dict
    hdr = Header.fromstring(hdrstr,sep='\n')

    # Now we read the catalog
    observations = fits['LDAC_OBJECTS'].read()

    # The image header gives us access to image-level quantities, like
    # EXPNUM, CCDNUM, MJD-OBS, etc. Careful, these quantities may have a different byte order than the catalog data. 
    EXPNUM = np.tile(hdr['EXPNUM'],len(observations))
    CCDNUM = np.tile(hdr['CCDNUM'],len(observations))

    # We can then append those quantities to the object array
    observations = rec_append_fields(observations,
                                      names=['EXPNUM','CCDNUM'],
                                      data=[EXPNUM,CCDNUM])
    return observations

    
class Matcher(Task):

    def __init__(self, receiver, session, input_path):
        Task.__init__(self)
        self.Receiver = receiver
        self.Session = session
        self.InputPath = input_path
    
    def match_file(self, session, path):

        with T["fits_read"]:
                input_data = read_bliss_file(path)
        with T["hpix"]:
                hpix = hp.ang2pix(nside=16384,theta=input_data['ALPHAWIN_J2000'],phi=input_data['DELTAWIN_J2000'],
                        lonlat=True, nest=True)

        input_data = rec_append_fields(input_data, "HPIX", hpix)
        input_data = np.sort(input_data, order="HPIX")
        min_hpix, max_hpix = input_data["HPIX"][0], input_data["HPIX"][-1]
        #print input_data["HPIX"][:100]

        job_input = np.array(zip(input_data['ALPHAWIN_J2000'], input_data['DELTAWIN_J2000'], input_data['HPIX']),
            dtype=[('ALPHAWIN_J2000',float),('DELTAWIN_J2000',float),('HPIX',int)]
        )
        matches = []
        

        class Callback:

            def on_streams_update(self, nevents, data):
                if "matches" in data:
                    for m in data["matches"]:
                        matches.append(m)
                        #for obs_i, cat_id, rgid, best_dist in m:
                        #    print "Match:      index: %10d    oject id: %10d      RG: %d     Dist:%f" % (obs_i, cat_id, rgid, best_dist)
                        #print "rgids:", m["rgid"], "oids:", m["object_id"]
                if "message" in data:
                    for msg in data["message"]:
                        print msg

            def on_exception(self, wid, info):
                print "Worker exception:\n--------------------"
                print info
                print "--------------------"                   

	    def on_job_finish(self, nevents, error):
		if error is not None:
			print "Job failed: %s" % (error,)


        job = session.createJob("Bliss",     #"Bliss", 
                                    use_data_cache = False,
                                    user_callback = Callback(),
                                    worker_class_file="match_worker.py",
                                    frame_selector = ("and", ("le", "hpix_min", max_hpix), ("ge", "hpix_max", min_hpix)),
                                    user_params = {"observations":job_input})
        with T["job"]:
                job.run()
        runtime = job.TFinish - job.TStart
        catalog_objects = job.EventsProcessed
        #print "Compared %d observations against %d catalog objects, elapsed time=%f" % (len(input_data), catalog_objects, runtime)


        mask = np.zeros((len(input_data),), dtype=np.bool)
        matched = []
        if matches:
            matches = np.concatenate(matches, axis=0)
            matches = np.sort(matches, order="index")
            
            #print "filtering: matches rgid range:", min(matches["rgid"]),  max(matches["rgid"])
            
            # prune matches so that only the best distance object remains for each observation
            dist = matches["best_dist"]
            oid = matches["object_id"]
            index = matches["index"]
            rgid = matches["rgid"]
            fileterd_matches = []
            nmatches = len(matches)
            i = 0
            while i < nmatches:
                best_dist = dist[i]
                best_oid = oid[i]
                best_rgid = rgid[i]
                inx = index[i]
                j = i + 1
                while j < nmatches and index[j] == inx:
                    #print "index %d repeated" % (inx,)
                    if dist[j] < best_dist:
                        best_dist = dist[j]
                        best_oid = oid[j]
                        best_rgid = rgid[j]
                    j += 1
                #print "Appending inx", inx
                fileterd_matches.append((inx, best_oid, best_rgid, best_dist))
                i = j
            matches = np.array(fileterd_matches, dtype=[("index",np.int64),("object_id",np.int64),("rgid",np.int64),("best_dist",np.float64)])
            mask[matches["index"]] = True
            matched = input_data[mask]
            #print "point_1: matches rgid range:", min(matches["rgid"]),  max(matches["rgid"])
            matched = rec_append_fields(matched, names=["OBJECT_ID","rgid"], data=(matches["object_id"], matches["rgid"]))
            #print "after filtering: matched rgid range:", min(matched["rgid"]),  max(matched["rgid"])
            #print "len(matched)", len(matched)
            #print len(matched)
        return len(input_data), catalog_objects, matched, input_data[mask==False]
        
    def run(self):
        print "started %s" % (self.InputPath,)
        n_observations, n_objects, matches, unmatches = self.match_file(self.Session, self.InputPath)
        self.Receiver.addData(self, n_observations, n_objects, matches, unmatches)

class MatchJob(Primitive):
    def __init__(self, session, files, max_matchers, stagger):
        Primitive.__init__(self)
        self.Matches = []
        self.Unmatches = []
        self.Queue = TaskQueue(max_matchers, stagger=stagger,
            tasks = [Matcher(self, session, path) for path in files]
        )
        
    def wait(self):
        self.Queue.waitUntilEmpty()
        
    @synchronized
    def addData(self, matcher, n_observations, n_objects, matches, unmatches):
        print "File: %s: %s observations comapred to %d objects: %d matches, %d unmatches" % (
                matcher.InputPath, n_observations, n_objects, len(matches), len(unmatches))
        if len(matches):
            self.Matches.append(matches)
        if len(unmatches):
            self.Unmatches.append(unmatches)
        

    
Usage = """
python match_exposure.py <output_prefix> <input file> ...
"""

opts, args = getopt.getopt(sys.argv[1:], "?hm:s:")
opts = dict(opts)
max_matchers = int(opts.get("-m", 5))
stagger = float(opts.get("-s", 0.1))

if len(args) < 2 or "-?" in opts or "-h" in opts:
    print Usage
    sys.exit(1)

outprefix = args[0]
files = args[1:]

#job_server_address = ("dbwebdev.fnal.gov", 8765) 
job_server_address = ("ifdb01.fnal.gov", 8765) 
session = Session(job_server_address)		#, worker_tags=["DES"])

job = MatchJob(session, files, max_matchers, stagger)
job.wait()

all_matches = job.Matches
all_unmatches = job.Unmatches

if len(all_matches):
    all_matches = np.concatenate(all_matches)
    oidmap = {}
    for i in xrange(len(all_matches)):
        rgid = all_matches["rgid"][i]
        oid = all_matches["OBJECT_ID"][i]
        range = oidmap.get(rgid, (oid, oid))
        omin, omax = range
        oidmap[rgid] = (min(oid, omin), max(oid, omax))
    #print oidmap
    print "Saving %d matches" % (len(all_matches),) 
    #print "oid range:", min(all_matches["OBJECT_ID"]), max(all_matches["OBJECT_ID"])
    #print "rgid range:", min(all_matches["rgid"]), max(all_matches["rgid"])
    fitsio.write(outprefix+"_matches.fits", all_matches, clobber=True)

if len(all_unmatches):
    all_unmatches = np.concatenate(all_unmatches)
    np.sort(all_unmatches, order="HPIX")
    print "Saving %d unmatches" % (len(all_unmatches),)
    fitsio.write(outprefix+"_unmatches.fits", all_unmatches, clobber=True)







