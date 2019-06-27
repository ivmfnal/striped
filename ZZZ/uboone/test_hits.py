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
from trace import Tracer




#columns = ["evtwgt.weight", "evtwgt_funcname", "evtwgt_nweight", "trkhits_pmtrack.dedx", "trkhits_pmtrack.xyz", "track_pmtrack.trkpdgtruth_pmtrack",
#    "mcevts_truth", "mcshwr_AncesotorProcess", "beamtime"]
T = Tracer(calibrate=True)

url = sys.argv[1]
dataset_name = sys.argv[2]

client = StripedClient(url)

T = Tracer()

dataset = Dataset(client, dataset_name, columns, trace=T)



for ie, e in enumerate(dataset.events()):

    hitsxyz = e.segment("trkhits_pmtrack.xyz")
    #print type(hitsxyz), hitsxyz.shape, hitsxyz.dtype, hitsxyz.base is None



    
T.printStats()

