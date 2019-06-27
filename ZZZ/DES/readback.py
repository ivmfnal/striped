import os, time, itertools
import sys, getopt, json, pprint
from couchbase import FMT_BYTES, FMT_JSON
from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError
import numpy as np
from numpy.lib.recfunctions import append_fields
import uproot
from striped.client import StripedClient
import fitsio, healpy

opts, args = getopt.getopt(sys.argv[1:], "")
data_url = args[0]
dataset = args[1]
rgid = int(args[2])
columns = args[3:]

c = StripedClient(data_url)
ds = c.dataset(dataset)

for cn in columns:
    c = ds.column(cn)
    print "column %s: %s" % (cn, c.descriptor)
    
print "RGInfo:", ds.rginfo(rgid)

for cn in columns:
    stripe = ds.stripe(cn, rgid)
    print "Data %s: %s %s %s" % (cn, stripe.dtype, stripe.shape, stripe) 
