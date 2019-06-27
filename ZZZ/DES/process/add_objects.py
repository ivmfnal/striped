from striped.client import CouchBaseBackend

import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import fitsio, healpy as hp
from astropy.io.fits import Header

class ProvenanceSegment:

        def __init__(self, file_name, begin_event = 0, nevents = 0):
                self.FileName = file_name
                self.BeginEvent = begin_event
                self.NEvents = nevents

        def toDict(self):
                return dict(
                        FileName        =       self.FileName,
                        BeginEvent      =       self.BeginEvent,
                        NEvents         =       self.NEvents
                )

class RGInfo:

        Version = "3.0"

        def __init__(self, rgid, nevents=None, segment=None, profile = None):
                assert nevents is not None or segment is not None
                self.RGID = rgid
                self.NEvents = nevents if nevents is not None else segment.NEvents
                self.Provenance = [segment] if segment is not None else []
                self.Profile = profile

        def toDict(self):
                return dict(
                        _Version     = self.Version,
                        RGID         = self.RGID,
                        NEvents      = self.NEvents,
                        Segments     = [s.toDict() for s in self.Provenance],
                        Profile      = self.Profile
                )

def add_objects(backend, input_data, dataset_name, rg_size):
    StripeHeaderFormatVersion = "1.0"

    schema = backend.schema(dataset_name)
    
    nobjects = len(input_data)
    counter_key = "%s:@bliss_next_object_id" % (dataset_name,)
    try:    
        backend.counter(counter_key)             # if the counter does not exist
    except:
        backend.counter(counter_key, initial=1)  # ... create is
        print "Counter bliss_next_object_id created"
    
    first_oid = backend.counter(counter_key, delta=nobjects).value-nobjects
    #print "first_oid=", first_oid

    input_data.sort(order="HPIX")
    
    object_id = np.arange(nobjects, dtype=np.int64) + first_oid
    bad = np.zeros((nobjects,), dtype=np.int8)
    input_data = rec_append_fields(input_data, names=["OBJECT_ID","BAD"], data=[object_id, bad])
    
    
    ngroups = max(nobjects//rg_size, 1)
    group_size = int(float(nobjects)/ngroups+0.5)
    rgid =  backend.counter("%s:@@nextRGID" % (dataset,), delta=ngroups).value - ngroups
    
    observation_columns = schema["branches"]["Observation"].keys()
    object_columns = schema["attributes"].keys()
    
    for i in xrange(0, nobjects, group_size):
    
        # Create objects
    
        nobjects_this_group = min(nobjects-i, group_size)
        object_stripes = {}
        
        for cn, cdesc in schema["attributes"].items():
            array = input_data[cn][i:i+group_size].copy()
            array = np.asarray(array, cdesc["dtype"])
            object_stripes[cn] = array
        
        oid_min, oid_max = min(object_stripes["OBJECT_ID"]), max(object_stripes["OBJECT_ID"])
        hp_min, hp_max = min(object_stripes["HPIX"]), max(object_stripes["HPIX"])

        rginfo = RGInfo(rgid, nevents=nobjects_this_group, 
                profile=dict(
                    hpix_min = hp_min, hpix_max = hp_max,
                    object_id_min = oid_min, object_id_max = oid_max
                )
        )
        
        key = "%s:@@rginfo:%s.json" % (dataset_name, rgid)

        backend[key].json = rginfo.toDict()

        backend.put_arrays(dataset_name, [(rgid, cn, stripe) for
                cn, stripe in object_stripes.items()])
        
        # Add first observations
                
        obs_stripes = {"Observation.@size": np.ones((nobjects_this_group,), dtype=np.int64)}        # one observation per object
        
        for cn, cdesc in schema["branches"]["Observation"].items():
            array = input_data[cn][i:i+group_size].copy()
            array = np.asarray(array, cdesc["dtype"])
            obs_stripes["Observation."+cn] = array

        backend.put_arrays(dataset_name, [(rgid, key, stripe) for key, stripe in obs_stripes.items()])
        
        
        print "RG %d stored with %d objects, pix range: %d:%d, object ids: %d:%d" % (rgid, group_size, 
                hp_min, hp_max, oid_min, oid_max)
        rgid += 1
    
            
if __name__ == "__main__":
    import sys, time, getopt
    
    Usage = """
    python add_objects.py [options] <bucket name> <dataset name> <file.fits>
    options:
        -n <target row group size> - default 100000
        -c <couchbase config>   - default environment COUCHBASE_BACKEND_CFG 
        -i - reinitialize the object id counter so that the next object will be given oject id = 1
    """

    
    opts, args = getopt.getopt(sys.argv[1:], "n:h?c:i")
    opts = dict(opts)
    if '-h' in opts or '-?' in opts or len(args) != 3:
        print Usage
        sys.exit(1)
    
    init_oid = "-i" in opts    
    config = opts.get("-c")
    group_size = int(opts.get("-n", 100000))

    bucket, dataset, path = args
    
    data = fitsio.read(path)
    print "%d objects in the input file %s" % (len(data), path)
    backend = CouchBaseBackend(bucket)
    
    if init_oid:
        counter_key = "%s:@bliss_next_object_id" % (dataset,)
        try:    
            backend.delete([counter_key])             # remove if exists
        except:
            pass
        backend.counter(counter_key, initial=1)
        print "Counter bliss_next_object_id initialized to 1"
         
    
    add_objects(backend, data, dataset, group_size)
        
    
    
