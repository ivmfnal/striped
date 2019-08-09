import time, sys, random

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

try:
    # if couchbase is unavailable, ignore until these names are used
    import couchbase
    from couchbase.bucket import Bucket
    from couchbase.cluster import Cluster, PasswordAuthenticator
    from couchbase.views.iterator import View
    from couchbase.views.params import Query
    from couchbase.exceptions import KeyExistsError, TemporaryFailError, TimeoutError, NotFoundError, TooBigError, NoMemoryError
except:
    pass

if PY3:
	from configparser import ConfigParser
else:
	from ConfigParser import ConfigParser
import os
import numpy as np
from striped.common import stripe_key, rginfo_key, standalone_data_key, stripe_header, data_header, parse_data
from striped.common.exceptions import StripedNotFoundException

class OpTimer:

    def __enter__(self):
        self.T0 = time.time()
        return self
        
    def __exit__(self, *args):
        self.T = time.time() - self.T0
        
    def __str__(self):
        return "%f" % (self.T,)

class   CouchBaseConfig:

    SectionName = "CouchBase"

    def __init__(self, config = None, path=None, envVar=None):
        # config is supposed to be ConfigParser object after the .read() is done
        if config is None:
            config = ConfigParser()
            if not path:
                if envVar is None:
                    envVar = 'COUCHBASE_BACKEND_CFG'
                path = os.environ[envVar]
            config.read(path)
        self.Config = config

        self.Username = self.get("Username")
        self.Password = self.get("Password")
        self.ReadonlyUsername = self.get("Readonly_Username", self.Username)
        self.ReadonlyPassword = self.get("Readonly_Password", self.Password)
        self.DefaultBucket = self.get("DefaultBucket")
        self.BucketInfoURL = self.get("BucketInfoURL")
        
        self.ClusterURL = self.get("ClusterURL")
        assert self.ClusterURL is not None, "Cluster URL not found in the CouchBase configuration"
        
    def get(self, key, default=None):
        try:    return self.Config.get(self.SectionName, key)
        except: return default


def retry(method, retries=20):
    def retry_method(self, *params, **args):
        done = False
        sleep_time = 1.0
        rem_retries = retries
        while rem_retries > 0:
                try:    
                    return method(self, *params, **args) 
                except TimeoutError as e:
                    pass
                except TemporaryFailError as e:
                    pass
                except NoMemoryError as e:
                    pass
                    
                if self.PrintErrors and rem_retries < retries//2:
                        print(("_upsert_and_retry:%s\n    remaining retry count: %d     sleep time: %.1f seconds" % (e, rem_retries, sleep_time)))
                time.sleep((1.0+random.random())/2*sleep_time)
                sleep_time *= 1.5
                rem_retries -= 1
        raise ValueError("Number of retries exceeded. DB insert operation failed")
    return retry_method
        
    
class CouchBaseBackend(object):

    MAX_SIZE = 15000000 - 1     # < 20MB
    GET_PUT_CHUNK = 20

    def __init__(self, bucket_name, cluster_url = None, username = None, password = None, read_only = False, 
                    config = None, print_errors = False,
                    **bucket_options):
        if isinstance(config, str) or config is None:
            try:    config = CouchBaseConfig(path = config)
            except: config = None
        elif isinstance(config, ConfigParser):
            config = CouchBaseConfig(config = config)
        if config is not None:
            if not read_only:   username, password = config.Username, config.Password
            else: username, password = config.ReadOnlyUsername, config.ReadOnlyPassword
            cluster_url = config.ClusterURL
        self.PrintErrors = print_errors
        self.Cluster = Cluster(cluster_url)
        self.Cluster.authenticate(PasswordAuthenticator(username, password))
        if bucket_name is None:
            bucket_name = config.DefaultBucket
        self.BucketName = bucket_name
        self.CB = self.Cluster.open_bucket(bucket_name, **bucket_options)

    @property
    def bucket(self):
        return self.CB
        
    def slice(self, length, max_segment):
        n = length//max_segment + 1
        N = length
        k = N % n
        m = (N-k)//n
        i = 0
        out = [m+1]*k + [m]*(n-k)
        return out
        
    @retry
    def _upsert_single_key(self, key, value, format):
        return self.CB.upsert(key, value, format=format)
        
    def _upsert_and_retry(self, data, format, retries=20):

        opid = int(time.time()*1000.0) % 1000

        #
        # 1. Try to upsert the whole dictionary at once
        #

        try:
            self.CB.upsert_multi(data, format=format)
        except TooBigError as e:
            #print dir(e)
            key = e.key
            data_size = len(data[key])
            raise
        except TimeoutError as e:
                pass
        except TemporaryFailError as e:
                pass
        else:
                return

        #if self.PrintErrors:
        #       print "%d: bulk upsert of dictionary with size %d failed. Will insert values one by one" % (opid, len(data))

        #
        # If bulk upsert has failed, insert items one by one with retries
        #

        for key, value in data.items():
            try:    
                self._upsert_single_key(key, value, format)
            except TooBigError as e:
                #print dir(e)
                key = e.key
                data_size = len(data[key])
                raise
        #if self.PrintErrors:
        #       print "%d: upsert done" % (opid,) 

    def _put_multi(self, data_dict, format):
        if len(data_dict) > self.GET_PUT_CHUNK:
            keys = data_dict.keys()
            n = len(keys)
            for i in range(0, n, self.GET_PUT_CHUNK):
                chunk = { k:data_dict[k] for k in keys[i:i+self.GET_PUT_CHUNK] }
                #print "_put_multi: chunk=", len(chunk), "format=", format, 
                #print [(k,len(d)) for k, d in chunk.items()]
                self._upsert_and_retry(chunk, format)
                #print "done"
        else:
            with OpTimer() as t:
                self._upsert_and_retry(data_dict, format)
            #print len(data_dict), sum(map(len, data_dict.values())), t
            
    put = _put_multi

    @retry
    def ____get_multi(self, keys, no_format = False):
        out_data = {}
        if len(keys) > self.GET_PUT_CHUNK:
            for i in range(0, len(keys), self.GET_PUT_CHUNK):
                data = self.CB.get_multi(keys[i:i+self.GET_PUT_CHUNK], quiet=True, no_format=no_format)
                for k, v in data.items():
                    out_data[k] = v.value if v is not None else None
        else:
            data = self.CB.get_multi(keys, quiet=True, no_format=no_format)
            for k, v in data.items():
                out_data[k] = v.value if v is not None else None
        return out_data
        
    @retry
    def _get_multi(self, keys, no_format = False):
        out_data = {}
        for i in range(0, len(keys), self.GET_PUT_CHUNK):
            ksegment = keys[i:i+self.GET_PUT_CHUNK]
            data = self.CB.get_multi(ksegment, quiet=True, no_format=no_format)
            for k in ksegment:
                v = data.get(k)
                out_data[k] = v.value if v is not None else None
        return out_data
        
    get = _get_multi
        
    def put_json(self, data_dict):
        self._put_multi(data_dict, couchbase.FMT_JSON)
        
    def get_json(self, keys):
        return self._get_multi(keys, no_format=False)

    def put_data(self, data_dict):
    
        split_dict = {}
        to_delete = set()
        for k, v in data_dict.items():
            l = len(v)
            if l > self.MAX_SIZE:
                to_delete.add(k)
                sizes = self.slice(l, self.MAX_SIZE) + [0]      # 0 will signal end of the split when reading
                #print "splitting data (%d) for key %s: %s" % (l, k, sizes)
                j = 0
                for i, n in enumerate(sizes):
                    split_dict[k+".%d" % (i,)] = v[j:j+n]
                    j += n
            else:
                split_dict[k] = v
        #if len(split_dict) > len(data_dict):
        #    #print "split_dict size: %d" % (len(split_dict),)
        if to_delete:
                self.delete(list(to_delete))
        self._put_multi(split_dict, couchbase.FMT_BYTES)
        
    def put_standalone_data(self, dataset_name, data_dict):
        put_dict = { standalone_data_key(dataset_name, k) : d for k, d in data_dict.items() }
        return self.put_data(self, data_dict)
        

    def get_standalone_data(self, dataset_name, keys):
        keys = { k: standalone_data_key(dataset_name, k) for k in keys }
        data_dict = self.get_data(keys.values())
        # stop here
                
    def get_data(self, keys):
        #print "get_data(%s)" % (keys,)
        out = {}
        #print keys
        data = self._get_multi(keys, no_format=True)
        multiparts = []
        for key, value in data.items():
            if value is None:
                multiparts.append(key)
            else:
                out[key] = value
                
        for key in multiparts:
            i = 0
            n = 10
            done = False
            segments = []
            while not done:
                parts_keys = ["%s.%d" % (key, i+j) for j in range(n)]
                parts_data = self._get_multi(parts_keys, no_format=True)
                for k in parts_keys:
                    segment = parts_data[k]
                    if not segment:         # either not found or zero length
                        done = True
                        break
                    else:
                        segments.append(segment)
                i += n
            if segments:
                out[key] = "".join(segments)

        #print out        
        return out

    def stripe_header(self, array):
        return stripe_header(array)
        
    def stripe_key(self, dataset_name, column_name, rgid):
        return stripe_key(dataset_name, column_name, rgid)
    
    def put_arrays(self, dataset_name, data_list):
        #
        #  data_list: [(rgid, field_name, array),...]
        #
    
        data_dict = {}
        for rgid, cn, arr in data_list:
            header = self.stripeHeader(arr)
            key = self.stripe_key(dataset_name, cn, rgid)
            data_dict[key] = bytes(header) + bytes(arr.data)
        return self.put_data(data_dict)
        
    def get_arrays(self, dataset_name, rgid, column_names, dtypes={}):
        keys = {cn:"%s:%s:%d.bin" % (dataset_name, cn, rgid) for cn in column_names}
        data_dict = self.get_data(keys.values())
        out_dict = {}
        for cn in column_names:
            key = keys[cn]
            data = data_dict[key]
            if data is None:
                raise KeyError("Stripe for column %s (key=%s) not found" % (cn, key))
            dtype = None
            if data.startswith("#__header:"):
                i = data.index("#", 1)
                if i > 0:
                    header = data[10:i]
                    data = data[i+1:]
                    fields = header.split(";")
                    for f in fields:
                        if f.startswith("dtype="):
                            dtype = f.split("=",1)[1]
                            break
            if dtype is None:   
                print("dtype not found in header:", repr(data[:100]))
                dtype = dtypes[cn]
            out_dict[cn] = np.frombuffer(data, dtype)
        return out_dict
            
    def convertStripe(self, data, desc):
        dtype = desc.NPType
        if data[:10] == "#__header:":
            i = data.index("#", 1)
            if i > 0:
                header = data[10:i]
                data = data[i+1:]
                fields = header.split(";")
                for f in fields:
                    if f.startswith("dtype="):
                        dtype = f.split("=",1)[1]
                        break                
        data = np.frombuffer(data, dtype)
        if not desc.ConvertToNPType is None and desc.ConvertToNPType != desc.NPType:
            data = np.asarray(data, dtype=desc.ConvertToNPType)
        fs = desc.fixedShape
        if fs:
            data = data.reshape((-1,) + fs)
        return data
            


    class _KeyAccessor(object):
        def __init__(self, backend, key):
            self.Backend = backend
            self.Key = key

        def __get_data(self):
            return self.Backend.get_data([self.Key])[self.Key]
            
        def __put_data(self, data):
            return self.Backend.put_data({self.Key: data})
            
        data = property(__get_data, __put_data)

        def __get_json(self):
            return self.Backend.get_json([self.Key])[self.Key]
            
        def __put_json(self, data):
            return self.Backend.put_json({self.Key: data})
            
        json = property(__get_json, __put_json)
            
    def __getitem__(self, key):         
        #
        # This can be used like this:
        #
        # backend[key].data = bytes("xyz123")
        # print backend[key].data
        # backend[key].json = { "name": "cucumber", "color":"green" }
        # print backend[key].json["color"]
        #
        return self._KeyAccessor(self, key)
        
    def keys(self, dataset):
        q = Query()
        q.mapkey_single = dataset
        v = View(self.CB, "views", "keys", query=q)
        return (x.value for x in v)
        
    def columns(self, dataset = None):
        q = None
        if dataset != None:
            q = Query()
            q.mapkey_single = dataset
        v = View(self.CB, "views", "columns", query=q)
        return ((x.key, x.value) for x in v)

    def datasets(self):
        v = View(self.CB, "views", "schemas")
        datasets = set()
        for x in v:
            datasets.add(x.key)
        return list(datasets)
        
    def schema(self, dataset):
        try:    schema = self.CB.get("%s:@@schema.json" % (dataset,)).value
        except NotFoundError:
            raise StripedNotFoundException("Dataset %s not found in CouchBase bucket %s" % (dataset, self.BucketName))
        return schema
        
    def RGInfos(self, dataset = None):
        q = None
        if dataset != None:
            q = Query()
            q.mapkey_single = dataset
        v = View(self.CB, "views", "RGInfos", query=q)
        if dataset is None:
            return ((x.key, x.value) for x in v)
        else:
            return (x.value for x in v)

    def RGIDs(self, dataset = None):
        if dataset is None:
            return ((ds, info["RGID"]) for ds, info in self.RGInfos(dataset))
        else:
            return (info["RGID"] for info in self.RGInfos(dataset))

    @retry
    def counter(self, key, delta=0, **args):
        return self.CB.counter(key, delta=delta, **args)
        
    def allocateRGIDs(self, dataset_name, n):
            return self.counter("%s:@@nextRGID" % (dataset_name,), delta=n).value - n

    def RGMap(self, dataset_name):
        try:
            key = "%s:@@rgmap.json" % (dataset_name,)
            rgmap = self[key].json
        except:
            print("Can not get row group map from the database")
            raise
        return rgmap            # as dictionary

    def keys(self, dataset_name):
        q = Query()
        q.mapkey_single = dataset_name
        v = View(self.CB, "views", "keys", query=q)
        return (x.value for x in v)
        
    @retry
    def _del_multi(self, keys):
        assert isinstance(keys, list)
        self.CB.remove_multi(keys, quiet=True)
    
    def delete(self, keys, batch_size=500):
        if isinstance(keys, str):
            keys = [keys]
        group = []
        n = 0
        for k in keys:
            group.append(k)
            n += 1
            if len(group) >= batch_size:
                self._del_multi(group)
                group = []
        if group:
            self._del_multi(group)
        return n
        
    __delitem__ = delete
    
    def putStripes(self, dataset, rgid, stripes):
        data_dict = {"%s:%s:%d.bin" % (dataset, sn, rgid):data
                for sn, data in stripes.items()
        }
        self.put_data(data_dict)
        
    def getStripes(self, dataset, rgid, columns):
        keys = ["%s:%s:%d.bin" % (dataset, cn, rgid) for cn in columns]
        return self.get_data(keys)

    def putStandaloneData(self, data_dict, dataset_name=None, as_json = False):
        if as_json:
            return self._put_standalone_json(data_dict, dataset_name)
        put_dict = {}
        for k, data in data_dict.items():
            hdr = data_header(data)
            if isinstance(data, np.ndarray):
                data = bytes(hdr) + bytes(data.data)
            else:
                data = bytes(hdr) + bytes(data)
            put_dict[standalone_data_key(k, dataset_name)] = data
        return self.put_data(put_dict)
        
    def getStandaloneData(self, keys, dataset_name=None, as_json = False):
        if as_json:
            return self._get_standalone_json(keys, dataset_name)
        db_keys = { k: standalone_data_key(k, dataset_name) for k in keys }
        data_dict = self.get_data(db_keys.values())
        out_dict = { k: parse_data(data_dict.get(db_keys[k])) for k in keys }
        return out_dict
                
    def _put_standalone_json(self, data_dict, dataset_name):
        put_dict = {}
        for k, data in data_dict.items():
            put_dict[standalone_data_key(k, dataset_name, json=True)] = data
        return self.put_json(put_dict)
        
    def _get_standalone_json(self, keys, dataset_name):
        db_keys = { k: standalone_data_key(k, dataset_name, json=True) for k in keys }
        data_dict = self.get_json(db_keys.values())
        out_dict = { k: data_dict[db_keys[k]] for k in keys }
        return out_dict
                
if __name__ == '__main__':
    import sys

    bucket_name = sys.argv[1]
    
    b = CouchBaseBackend(bucket_name)
    
    def test_xlarge_read_write():

        data = 'abcd'
        prefix = 'key_'

        items = {}

        while len(data) < CouchBaseBackend.MAX_SIZE * 10:
            key = prefix + str(len(data))
            items[key] = data
            data = data + data + data
            data = data[:len(data)//2] + "12345"

        b.put_data(items)

        out_data = b.get_data(items.keys())

        for k in items.keys():
            if out_data[k] != items[k]:
                print(("Difference for key %s" % (k,)))
                
    def test_read_write():
        fixed = ":" + "1234567890"*10000
        
        data = {"key:%d" % (x,): str(x)+fixed for x in range(1000)}
        b.put_data(data)
        read_data = b.get_data(data.keys())
        diff = 0
        for k in data.keys():
            if data[k] != read_data[k]:
                print(("Data differs for key <%s>" % (k,)))
                diff += 1
        if not diff:
            print ("Data varification succeeded")
            
            
    def test_keys():
        it = b.keys(sys.argv[2])
        for k in it:
            print (k)   
            
    def test_rginfos():
        it = b.RGInfos(sys.argv[2])
        for v in it:
            print (v)   
            
    def test_rgids():
        it = b.RGIDs(sys.argv[2])
        for v in it:
            print((type(v)))
            print (v)   
            
    def test_schema():
        print((b.schema(sys.argv[2])))
            
    test_read_write()
    
