from WSGIApp import WSGIApp, WSGIHandler, Application, Response
import couchbase
from ConfigParser import ConfigParser
from threading import RLock
import os, time, json, urllib2, zlib, yaml, urllib
from cStringIO import StringIO

from striped.client import CouchBaseConfig, CouchBaseBackend
from striped.common import standalone_data_key



def debug(msg):
    open("/tmp/striped.log","a").write(
            "%s: %s\n" % (time.ctime(time.time()), msg))


class   StripedConfiguration:
    def __init__(self, req, path=None, envVar=None):
        """
        path is a fully qualified path/file name of the
        configuration file to open.  If not provided it will
        will be taken from the environment varable envVar.  If no
        environment variable is passed in APACHE_PYTHON_CFG_FILE
        will be used.  If both path and envVar are defined, envVar
        will be used.
        """
        if not path:
            if envVar is None:
                envVar = 'STRIPED_SERVER_CFG'
            path = req.environ.get(envVar, None) or \
                os.environ[envVar]
                
        self.Config = yaml.load(open(path, "r").read())
        
        server_cfg = self.Config.get("Server", {})
        self.BucketInfoURL = server_cfg["bucket_info_url"]
        self.Buckets = sorted(server_cfg["buckets"])
        self.OperationTimeout = float(server_cfg.get("operation_timeout", 10.0))
        self.DataTTL = server_cfg.get("DataTTL", 30*24*3600)
        self.MetadataTTL = server_cfg.get("MetadataTTL", 24*3600)
        
        self.DatasetsToBuckets = self.Config.get("Datasets", {})       # dataset name -> bucket URL
        
        self.TagsConnect = self.Config.get("Tags", {}).get("connect")
        self.DefaultBucket = server_cfg.get("DefaultBucket")

    def bucketURLForDataset(self, dataset):
        bucket = self.DatasetsToBuckets[dataset]
        #debug("cfg: bucket=%s" % (bucket,))
        return self.ServerURL + "/" + bucket + "?operation_timeout=%f" % (self.OperationTimeout,)
        
    def bucketForDataset(self, dataset):
        return self.DatasetsToBuckets[dataset]
        
    #def bucketURL(self, bucket):
    #    return self.ServerURL + "/" + bucket + "?operation_timeout=%f" % (self.OperationTimeout,)
        
    def datasets(self):
        return self.DatasetsToBuckets.keys()
        
    def buckets(self):
        return self.Buckets

def cacheability_control(method):
    def decorated(*params, **args):
        resp = method(*params, **args)
        if args.get("cache","yes") != "yes":
            resp.headers["Cache-control"] = "no-store"
        return resp
    return decorated
            


class StripedHandler(WSGIHandler):

    def __init__(self, *params, **args):
        WSGIHandler.__init__(self, *params, **args)
        self.MetadataTTL = Config.MetadataTTL
        self.DataTTL = Config.DataTTL

    def hello(self, req, relpath, **args):
        return Response("hello")

    def index(self, req, relpath, **args):
        return self.render_to_response("index.html")
        
    def datasets(self, req, relpath, **args):
        ds_last = {}
        buckets = Config.buckets()
        ds_info = {}
        ds_to_bucket = {}
        for b in buckets:
            cb = self.App.backendForBucket(b)
            if cb is not None:
		for ds in cb.datasets():
			ds_to_bucket[ds] = b
		    	ne, ng, fileset = 0,0,set()
			for rginfo in cb.RGInfos(ds):
			    ne += rginfo["NEvents"]
			    ng += 1
			    for s in rginfo["Segments"]:
				fileset.add(s["FileName"])
			ds_info[ds] = (ne, ng, fileset)
        total_datasets = len(ds_info)
        total_events = sum([ne     for ne, _, _ in ds_info.values()])
        total_groups = sum([ng     for _, ng, _ in ds_info.values()])
        total_files = sum([len(fs) for _, _, fs in ds_info.values()])
        
        resp = self.render_to_response("datasets.html", 
            total_datasets = total_datasets, total_events = total_events, total_groups = total_groups,
            total_files = total_files,
            data=sorted([(ds, len(fs), ne, ng, int(float(ne)/ng+0.5) if ng > 0 else 0, ds_to_bucket[ds]) for ds, (ne, ng, fs) in ds_info.items()]
            ))
        resp.cache_expires(self.MetadataTTL)
        return resp
        
    def dataset_info(self, req, relpath, ds=None, **args):   
        backend = self.App.backendForDataset(ds)
        if backend is None:    
            return Response("Dataset %s is not found" % (ds,), status=400) 
        ne = 0
        ng = 0
        files = set()
        for rginfo in backend.RGInfos(ds):
            ne += rginfo["NEvents"]
            ng += 1
            for s in rginfo["Segments"]:
                files.add(s["FileName"])
        schema = backend.schema(ds)
        branches = [ (bname, sorted(bdict.items())) for bname, bdict in schema["branches"].items() ]
        return self.render_to_response("dataset_info.html", ds=ds, nfiles = len(files), nevents = ne, ngroups = ng,
                attributes = schema["attributes"].items(),
                branches = branches,
                )
        
    def buckets(self, req, relpath, **args):
        data = urllib2.urlopen(Config.BucketInfoURL).read()
        data = json.loads(data)
        data_lst = []
        my_buckets = Config.buckets()
        for binfo in data:
            bname = binfo["name"]
            if bname in my_buckets:
                basic_stats = binfo["basicStats"]
                memUsed = basic_stats["memUsed"]
                diskUsed = basic_stats["diskUsed"]
                items = basic_stats["itemCount"]
                ops = basic_stats["opsPerSec"]
                quotaUsed = basic_stats["quotaPercentUsed"]
                datasets = [ds for ds in Config.datasets() 
                                if Config.bucketForDataset(ds) == bname]
                data_lst.append((bname, items, float(diskUsed)/1024/1024/1024, 
                                    float(memUsed)/1024/1024/1024, quotaUsed, ops, 
                                    sorted(datasets)))
        data_lst.sort()     # by bname
        return self.render_to_response("buckets.html", data = data_lst)

    #
    # Data
    #

            
    def dataset_list(self, req, relpath, **args):
        lst = sorted(Config.datasets())
        return Response(json.dumps(lst), content_type="text/json")
        
    def dataset_schema(self, req, relpath, ds=None, **args):
        backend = self.App.backendForDataset(ds)
        schema = backend.schema(ds)
        return Response(json.dumps(schema), content_type="text/json")
        
    @cacheability_control
    def stripe(self, req, relpath, ds=None, rgid=None, column=None, compressed="no", **args):
        #
        # stripe?ds=<dataset>&rgid=<rgid>&column=<column>
        #
        compressed = compressed == "yes"
        backend = self.App.backendForDataset(ds)
        key = "%s:%s:%s.bin" % (ds, column, rgid)
        data = backend.get_data([key]).get(key)
        if data is None:
            return Response("Stripe %s %s %s not found" % (ds, column, rgid),
                    status=400)
        if compressed:
            data = zlib.compress(data, 1)   
        return Response(data, cache_expires=self.DataTTL)
        
    @cacheability_control
    def data(self, req, relpath, ds=None, key=None, compressed="no", json="no", **args):
        ds = urllib.unquote(ds or "")
        backend = self.App.backendForDataset(ds or self.App.defaultBucket())
        key = urllib.unquote(key)
        json = json == "yes"
        dbkey = standalone_data_key(ds, key, json=json)
        data = backend.get_data([dbkey]).get(dbkey)
        if data is None:
            return Response("Not found", status=400)
        if compressed:
            data = zlib.compress(data, 1)   
        return Response(data, cache_expires=self.DataTTL)
        
                         
    @cacheability_control
    def stripe_meta(self, req, relpath, ds=None, rgid=None, column=None, compressed="no", **args):
        #
        # stripe?ds=<dataset>&rgid=<rgid>&column=<column>
        #
        compressed = compressed == "yes"
        backend = self.App.backendForDataset(ds)
        key = "%s:%s:%s.bin" % (ds, column, rgid)
        data = backend.get_data([key]).get(key)
        if data is None:
            return Response("Stripe %s %s %s not found" % (ds, column, rgid),
                    status=400)
        if compressed:
            data = zlib.compress(data, 1)   
        return Response(json.dumps({"length": len(data)}), cache_expires=self.MetadataTTL, content_type="text/json")

    def update_stripes(self, req, relpath, ds=None, rgid=None, **args):
        data = req.body
        header, data = data.split("\n",1)
        l = len(data)
        i = 0
        stripes = {}
        for w in header.split():
            column, size = w.split(":")
            size = int(size)
            stripe = data[i:i+size]
            i += size
            assert i <= l       # i can be == l only after last stripe
            key = "%s:%s:%s.bin" % (ds, column, rgid)
            stripes[key] = stripe
                
        backend = self.App.backendForDataset(ds)
        backend.put_data(stripes)    
        return Response("OK")

    @cacheability_control
    def stripes(self, req, relpath, ds=None, rgid=None, columns=None, compressed="no", **args):
        compressed = compressed == "yes"
        backend = self.App.backendForDataset(ds)
        columns = columns.split(",")
        keys = ["%s:%s:%s.bin" % (ds, column, rgid) for column in columns]
        
        data = backend.get_data(keys)
        data_out = []
        for k, val in data.items():
            cn = k.split(":")[1]
            if val is not None:
                if compressed:
                    val = zlib.compress(val, 1)                
                data_out.append((cn, len(val), val))
        header = " ".join(("%s:%d" % (cn, n) for cn, n, v in data_out)) + "\n"
        
        def data_iterator(header, data):
            yield header
            for cn, n, v in data:
                yield v
                
        return Response(app_iter=data_iterator(header, data_out), cache_expires=self.DataTTL)
    
    @cacheability_control
    def stripes_sizes(self, req, relpath, ds=None, rgids=None, columns=None, **args):
    
        if req.body:
            params = json.loads(req.body)
            ds = params["ds"]
            rgids = params["rgids"]
            columns = params["columns"]
        else:
            rgids = rgids.split(",")
            columns = columns.split(",")
        backend = self.App.backendForDataset(ds)
        out = {}
        
        for cn in columns:
            keys = ["%s:%s:%s.bin" % (ds, cn, rgid) for rgid in rgids]
            data = backend.get_data(keys)
            lst = []
            for k in keys:
                rgid = int(k.split(".bin",1)[0].split(":")[2])
                lst.append([rgid, len(data[k])])
            out[cn] = lst
        out = json.dumps(out)
        exp = 0 if req.body else self.DataTTL
        return Response(out, cache_expires=exp, content_type="text/json")                
        
    def column_desc(self, req, relpath, ds=None, column=None, **args):
        backend = self.App.backendForDataset(ds)
        key = "%s:%s:@@desc.json" % (ds, column)
        data = backend.get_json([key]).get(key)
        if data is None:
            return Response("Column %s %s not found" % (ds, column), status=400)
        if not isinstance(data, (str, unicode)):
            data = json.dumps(data)
        return Response(data, cache_expires=self.MetadataTTL, content_type="text/json")                
        
    def column_descs(self, req, relpath, ds=None, columns=None, **args):
        backend = self.App.backendForDataset(ds)
        columns = columns.split(",")
        keys = ["%s:%s:@@desc.json" % (ds, column) for column in columns]
        data = backend.get_json(keys)
        out = {}
        for key, desc in data.items():
            if isinstance(desc, (str, unicode)):
                desc = json.loads(desc)
            column = key.split(":")[1]
            out[column] = desc
        return Response(json.dumps(out), content_type="text/json", cache_expires=self.MetadataTTL)
        
    def columns(self, req, relpath, ds=None, **args):
        backend = self.App.backendForDataset(ds)
        columns = backend.columns(ds)
        if ds is not None:
            columns = [cn for _, cn in columns]
        return Response(json.dumps(sorted(columns)), content_type="text/json", cache_expires=self.MetadataTTL)
        
    @cacheability_control
    def rgids(self, req, relpath, ds=None, **args):
        backend = self.App.backendForDataset(ds)
        rgids = backend.RGIDs(ds)
        return Response(json.dumps(sorted(rgids)), content_type="text/json", cache_expires=self.MetadataTTL)

    def batchLines(self, lines, n=10000, delimiter=""):
        buf = []
        for l in lines:
            if len(buf) >= n:
                yield delimiter.join(buf)
                buf = []
            buf.append(l)
        if buf:
            yield delimiter.join(buf)

    def streamListAsJSON(self, lst):
        yield "["
        first = True
        for x in lst:
            x = json.dumps(x)
            if not first:
                yield "," + x
            else:
                yield x
            first = False
        yield "]"        

    def allRGInfos(self, ds):
        backend = self.App.backendForDataset(ds)
        rginfos = backend.RGInfos(ds)
        return Response(app_iter=self.batchLines(
                self.streamListAsJSON(
                    (rginfo for dataset, rginfo in rginfos)
                )
            ), 
            content_type="text/json", cache_expires=self.MetadataTTL
        )
        
    @cacheability_control
    def rginfo(self, req, relpath, ds=None, rgid=None, rgids=None, **args):
        backend = self.App.backendForDataset(ds)
        if rgids:
            rgids = rgids.split(",")
            lst = []
            for r in rgids:
                words = r.split(":", 1)
                if len(words) == 1:
                    lst.append(int(r))
                else:
                    lst += range(int(words[0]), int(words[1])+1) 
            rgids = lst
        else:
            rgids = [int(rgid)]
        keys = ["%s:@@rginfo:%s.json" % (ds, rgid) for rgid in rgids]
        data = backend.get_json(keys)
        out = []
        for rgid in rgids:
            k = "%s:@@rginfo:%s.json" % (ds, rgid)
            info = data.get(k)
            if info is not None:
                if isinstance(info, (str, unicode)):
                    info = json.loads(info)
                out.append(info)
        return Response(json.dumps(out), content_type="text/json", cache_expires=self.MetadataTTL)
        
    def tagged_events(self, req, relpath, ds=None, event_id_range=None, conditions=None, **args):
        wheres = []
        tables = []
        event_id_range = map(int, event_id_range.split(":"))
        for i, c in enumerate(conditions.split(",")):
            if ":" in c:
                words = c.split(":")
                tag = words[0]
                op = {
                    "eq":   "=",    "ne":   "!=",
                    "lt":   "<",    "le":   "<=",
                    "gt":   ">",    "ge":   ">="}[words[1]]
                value = words[2]
                wheres.append("""
                    t%d.name='%s' and t%d.value %s '%s' 
                        and t%d.dataset = '%s'
                        and t%d.event_id = tags.event_id""" % (i, tag, i, op, value, i, ds, i))
            else:
                wheres.append("""t%d.name='%s' 
                        and t%d.dataset = '%s'
                        and t%d.event_id = tags.event_id""" % (i, c, i, ds, i))
            tables.append("tags t%d" % (i,))
        conn = self.App.tagsDatabase()
        c = conn.cursor()
        sql = """select distinct tags.event_id from tags, %s 
                where tags.dataset = '%s' 
                    and tags.event_id >= %d and tags.event_id < %d
                and %s order by tags.event_id""" % (",".join(tables), ds, event_id_range[0], event_id_range[1],
                    " and ".join(wheres))
        c.execute(sql)
        lst = [tup[0] for tup in c.fetchall()]
        return Response(json.dumps(lst), content_type="text/json", cache_expires=self.MetadataTTL)
                    
                
        
class BackendKeeper:

    # Couchbase 5.x version

    def __init__(self, cbconfig, bucket_name, timeout=30):
        self.Timeout = timeout
        self.LastTouch = time.time()
        self.BucketName = bucket_name
        self.Backend = None
        self.CBConfig = cbconfig
                
    def backend(self):
        if self.Backend is None or self.stale():
            try: self.Backend = CouchBaseBackend(self.BucketName, config = self.CBConfig)
            except: return None
        self.LastTouch = time.time()
        #debug("bucket for %s: %s" % (self.BucketURL, id(self.Bucket)))
        return self.Backend
        
    def stale(self):
        return self.LastTouch < time.time() - self.Timeout
        
        
class CBConnectionPool:
    def __init__(self, cbconfig, connection_stale_timeout = 30):
        self.Lock = RLock()
        self.CBConfig = cbconfig
        self.Backends = {}           # bucket_name -> bucket backend
        self.StaleTimeout = connection_stale_timeout
        
    def backend(self, bucket):
        with self.Lock:
            if not bucket in self.Backends:
                self.Backends[bucket] = BackendKeeper(self.CBConfig, bucket, self.StaleTimeout)
            return self.Backends[bucket].backend()
            
    __getitem__ = backend
            
ConfigLock = RLock()

Config = None   
CouchBaseCfg = None 

TagsConnection = None

def tidy_shape(shape):
    items = [x if x else "*" for x in shape]
    return "(" + ", ".join(["%s" % (x,) for x in items]) + ")"

class StripedApp(WSGIApp):

    ConnectionPool = None

    @classmethod
    def initConnectionPool(cls, cbconfig):
        if cls.ConnectionPool is None:
            cls.ConnectionPool = CBConnectionPool(cbconfig)
    
    def __init__(self, request, root_class):
        global Config, CouchBaseCfg
        WSGIApp.__init__(self, request, root_class)
        #debug("script home = %s" % (self.ScriptHome,))
        with ConfigLock:
            if Config is None:
                Config = StripedConfiguration(request, envVar = 'STRIPED_SERVER_CFG')     
                CouchBaseCfg = CouchBaseConfig()
            if self.ConnectionPool is None:
                self.initConnectionPool(CouchBaseCfg)
                
        self.setJinjaFilters({
            "tidy_shape": tidy_shape
            })
        
    def scanDatasets(self):
        dataset_map = {}        # dataset name -> bucket name
        for bucket in Config.buckets():
            backend = self.backendForBucket(bucket)
            for ds in backend.datasets():
                dataset_map[ds] = bucket
        return dataset_map
        
    def backendForDataset(self, dataset):
        bucket_name = Config.bucketForDataset(dataset)
        #debug("app: bucket URL=%s" % (bucket_url,))
        return self.ConnectionPool[bucket_name]
        
    def backendForBucket(self, bucket_name):
        return self.ConnectionPool[bucket_name]
        
        
    def tagsDatabase(self):
        global TagsConnection
        if TagsConnection is None and Config.TagsConnect is not None:
            TagsConnection = psycopg2.connect(Config.TagsConnect)
        return TagsConnection
        
        
application = Application(StripedApp, StripedHandler)

        
if __name__ == '__main__':
    import sys
    
    cfg = StripedConfiguration(None, sys.argv[1])
    for ds in sorted(cfg.datasets()):
        print ds, cfg.bucketForDataset(ds)
