import fitsio, healpy as hp
import numpy as np
from numpy.lib.recfunctions import append_fields
import sys, getopt, random

Usage="""
python sample.py <fraction> <out_file> <in_file> ...
"""

fraction = float(sys.argv[1])
out_file = sys.argv[2]
files = sys.argv[3:]

out_array = []

for fn in files:
    array = fitsio.read(fn, columns=["RA","DEC","COADD_OBJECT_ID"])
    hpix = hp.ang2pix(nside=16384,theta=array['RA'],phi=array['DEC'],
        lonlat=True, nest=True)
    array = append_fields(array, "HPIX", hpix)
    inxrange = np.arange(len(array))    
    n = int(len(array)*fraction+0.5)
    inx = random.sample(inxrange, n)
    segment = array[inx]
    print len(segment), "observations from", fn
    out_array.append(segment)

out_array = np.concatenate(out_array)

out_array.sort(order="HPIX")

print len(out_array), out_array.dtype, out_array

fitsio.write(out_file, out_array, clobber=True)
    

    
    
    
    
