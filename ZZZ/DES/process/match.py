from striped.job import SinglePointStripedSession as Session
import numpy as np
from numpy.lib.recfunctions import append_fields
import fitsio, healpy as hp
import sys, time

from striped.common import Tracer
T = Tracer()

def match_file(session, path):

    with T["fits/read"]:
            input_data = fitsio.read(path, ext=2, columns=["ALPHAWIN_J2000","DELTAWIN_J2000"])
    with T["hpix"]:
            hpix = hp.ang2pix(nside=16384,theta=input_data['ALPHAWIN_J2000'],phi=input_data['DELTAWIN_J2000'],
                    lonlat=True, nest=True)

    hpix = np.asarray(hpix, np.float64)
    input_data = append_fields(input_data, "HPIX", hpix)
    np.sort(input_data, order="HPIX")

    input_data = np.array(zip(input_data['ALPHAWIN_J2000'], input_data['DELTAWIN_J2000'], input_data['HPIX']))
    matches = []

    class Callback:

        def on_streams_update(self, nevents, data):
            if "matches" in data:
                for m in data["matches"]:
                    matches.append(m)
                    for obs_i, cat_id, rgid, obs_ra, obs_dec, cat_ra, cat_dec in m:
                        print "Match:      index: %10d    RA: %9.4f    Dec: %9.4f   RG: %d" % (int(obs_i), obs_ra, obs_dec, rgid)
                        print "   COADD oject id: %10d        %9.4f         %9.4f" % (int(cat_id), cat_ra, cat_dec)
            if "message" in data:
                for msg in data["message"]:
                    print msg

        def on_exception(self, wid, info):
            print "Worker exception:\n--------------------"
            print info
            print "--------------------"                   


    job = session.createJob("Bliss",     #"Bliss", 
                                user_callback = Callback(),
                                worker_class_file="match_worker.py",
                                user_params = {"observations":input_data})
    with T["job"]:
            job.run()
    runtime = job.TFinish - job.TStart
    catalog_objects = job.EventsProcessed
    print "Compared %d observations against %d catalog objects, elapsed time=%f" % (len(input_data), catalog_objects, runtime)


    if matches:
        matches = np.concatenate(matches, axis=0)
    return matches



job_server_address = ("ifdb01.fnal.gov", 8765) #production
session = Session(job_server_address)

for path in sys.argv[1:]:
    matches = match_file(session, path)
    print path, matches







