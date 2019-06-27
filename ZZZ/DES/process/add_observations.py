from striped.client import CouchBaseBackend

import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import fitsio, healpy as hp
from astropy.io.fits import Header
from striped.common import Tracer

T = Tracer()

def dict_to_recarray(dct, keys=None):
    # cnmap : dct key -> column name in the resulting recarray

    with T["dict_to_recarray"]:
        keys = keys or dct.keys()
        types = [(k, dct[k].dtype, dct[k].shape[1:]) for k in keys] 

        l = len(dct[keys[0]])
        for k in keys:
            assert len(dct[k]) == l, "Column %s length %d != first column length %d" % (k, len(dct[k]), l)

        data = zip(*[dct[k] for k in keys])
        return np.array(data, types)
    
    
def recarray_to_dict(rec):
    with T["recarray_to_dict"]:
        cnames = rec.dtype.names
        out = {}
        for cn in cnames:
            out_name = cn
            out_arr = rec[cn]
            out[out_name] = np.ascontiguousarray(out_arr)
        return out

def add_observations(backend, in_observations_data, dataset_name):

    schema = backend.schema(dataset_name)
    obs_columns = [cn.encode("ascii", "ignore") for cn in schema["branches"]["Observation"].keys()]
    prefixed_obs_columns = ["Observation.%s" % (cn,) for cn in obs_columns]
    obs_attr_dtypes = {}
    for cn, desc in schema["branches"]["Observation"].items():
        cn= cn.encode("ascii", "ignore")
        dt = (desc["dtype"].encode("ascii","ignore"), tuple(desc["shape"]))
        obs_attr_dtypes[cn] = dt
    obs_attr_shapes = {cn:desc["shape"] for cn, desc in schema["branches"]["Observation"].items()}
    
    prefixed_to_cn = dict(zip(prefixed_obs_columns, obs_columns))
    cn_to_prefixed = dict(zip(obs_columns, prefixed_obs_columns))

    #print "-------in_observations_data_0.dtype=", in_observations_data["FLUXERR_APER"].dtype, in_observations_data["FLUXERR_APER"].shape
    in_observations_data = np.sort(in_observations_data, order=["rgid","OBJECT_ID"])
    in_rgids = in_observations_data["rgid"].copy()
    # apply dtypes according to the schema
    in_observations_data =  np.asarray(in_observations_data, list(obs_attr_dtypes.items()))       # rgid is not in the schema, so it will be removed here
    #print "-------in_observations_data_1.dtype=", in_observations_data["FLUXERR_APER"].dtype, in_observations_data["FLUXERR_APER"].shape

    # break the input data into dictionary of  by rgid and then by object id
    # note that this will only create views of arrays, without copying the contents
   
    # rename input columns to add Observation. prefix

    in_oids = in_observations_data["OBJECT_ID"].copy()
    in_observations_data.dtype.names = [cn_to_prefixed.get(cn,cn) for cn in in_observations_data.dtype.names] 

    with T["input_by_rgid"]:

        input_by_rgid = {}
        for irow in xrange(len(in_observations_data)):
            rgid, oid = in_rgids[irow], in_oids[irow]
            by_oid = input_by_rgid.setdefault(rgid, {})
            assert not oid in by_oid, "Warning: Found more than 1 observation for a single object %d in the exposire" % (oid,)
            by_oid[oid] = irow
    
    with T["rg_loop"]:
        # loop over all rg's found in the input data and update them

        for rgid, in_rg in input_by_rgid.items():

            with T["rg_loop/get_db_data"]:
                #
                # get observations and objects from the DB
                #
                db_observations_data = backend.get_arrays(dataset_name, rgid, prefixed_obs_columns)
                db_objects_data = backend.get_arrays(dataset_name, rgid, ["Observation.@size","OBJECT_ID"])
                nobs_column = db_objects_data["Observation.@size"]
                oid_column = db_objects_data["OBJECT_ID"]

            
            with T["rg_loop/reshape"]:
                # apply correct shape
                for pn in db_observations_data.keys():
                    if pn in prefixed_to_cn:
                        cn = prefixed_to_cn[pn]
                        shape = tuple(obs_attr_shapes[cn])
                        if len(shape):
                            db_observations_data[pn] = db_observations_data[pn].reshape((-1,)+shape)

            with T["rg_loop/convert"]:
                # convert db data into numpy record array
                db_observations_data = dict_to_recarray(db_observations_data, keys=in_observations_data.dtype.names)
                assert db_observations_data.dtype.names == in_observations_data.dtype.names
                
            
            with T["rg_loop/break"]:
                # break db data into segments by object id
                object_segments = {}
                merged = []
                new_size = nobs_column.copy()
                j = 0
                for i, (oid, old_size) in enumerate(zip(oid_column, nobs_column)):
                    if old_size > 0:
                        merged.append(db_observations_data[j:j+old_size])
                    if oid in in_rg:
                        irow = in_rg[oid]
                        merged.append(in_observations_data[irow:irow+1])
                        new_size[i] += 1
                    j += old_size
                out_observations_data = np.concatenate(merged)
                out_observations_data = recarray_to_dict(out_observations_data)
                out_observations_data["Observation.@size"] = new_size
            
            with T["rg_loop/put_arrays"]:
                backend.put_arrays(dataset_name, [(rgid, key, array) for key, array in 
                            out_observations_data.items()])            
                
        

if __name__ == "__main__":
    import sys, time, getopt
    
    Usage = """
    python add_observations.py [options] <bucket name> <dataset name> <matches_file.fits>
    options:
        -c <couchbase config>   - default environment COUCHBASE_BACKEND_CFG 
    """

    
    opts, args = getopt.getopt(sys.argv[1:], "h?c:")
    opts = dict(opts)
    if '-h' in opts or '-?' in opts or len(args) != 3:
        print Usage
        sys.exit(1)
    
    config = opts.get("-c")

    bucket, dataset, path = args
    
    data = fitsio.read(path)
    print "%d object-observation pairs in the input file %s" % (len(data), path)
    backend = CouchBaseBackend(bucket)
    
    add_observations(backend, data, dataset)
    
    T.printStats()
        
    
