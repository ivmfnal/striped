from trace import Tracer
from Q import Dataset
from Q import cosh, sqrt, cos
from StripedClient import StripedClient
import sys, math, traceback, time, os
import numpy as np

Usage = """
python test.py <data server url> <dataset name>
"""

columns = ["ntracks_pmtrack", "track_pmtrack.ntrkhits_pmtrack","trkhits_pmtrack.dedx", "trkhits_pmtrack.xyz"]

#columns = ["evtwgt.weight", "evtwgt_funcname", "evtwgt_nweight", "trkhits_pmtrack.dedx", "trkhits_pmtrack.xyz", "track_pmtrack.trkpdgtruth_pmtrack",
#    "mcevts_truth", "mcshwr_AncesotorProcess", "beamtime"]
T = Tracer(calibrate=True)

url = sys.argv[1]
dataset_name = sys.argv[2]

client = StripedClient(url)

dataset = Dataset(client, dataset_name, columns)

for ie, e in enumerate(dataset.events()):

    print "Event:", ie, "  N pmtracks:", e.ntracks_pmtrack, "  len(pmtrack):", len(e.track_pmtrack)
    j = 0
    for it, t in enumerate(e.track_pmtrack):
        print "  Track:", it, "  nhits:", t.ntrkhits_pmtrack
        for ip in (0,1,2):
            nh = max(t.ntrkhits_pmtrack[ip], 0)
            for ihit in xrange(nh):
                dedx = e.trkhits_pmtrack[j].dedx
                xyz = e.trkhits_pmtrack[j].xyz
                print "      Plane:",ip,"   Hit:",ihit,"   xyz:",xyz,"   dedx=",dedx
                j += 1             
    
    
    
    #"  Hits per plane:", e.track_pmtrack.ntrkhits_pmtrack

    continue
    
    print e.mcevts_truth, e.mcshwr_AncesotorProcess, e.beamtime, e.evtwgt_nweight

    print "weights=", e.evtwgt[:]
    
    for iw, w in enumerate(e.evtwgt):
        print ie, iw, w.weight

    for ifn, fn in enumerate(e.evtwgt_funcname):
        print ie, ifn, fn

    print len(e.trkhits_pmtrack)  
    for ihit, hit in enumerate(e.trkhits_pmtrack[:10]):
        print ie, ihit, hit.dedx, hit.xyz 

    #for trk in e.track_pmtrack:
    #    print trk.trkpdgtruth_pmtrack     

    
        

