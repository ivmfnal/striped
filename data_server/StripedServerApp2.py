from webpie import WebPieApp, WebPieHandler, Response, app_synchronized
import couchbase
from ConfigParser import ConfigParser
from threading import RLock
import os, time, json, urllib2, zlib, yaml, urllib, sys
from cStringIO import StringIO

from striped.client import CouchBaseConfig, CouchBaseBackend
from striped.common import standalone_data_key, stripe_key
from striped.common.signed_token import SignedToken, generate_secret, TokenBox
from striped.common.rfc2617 import digest_server
from striped.common.exceptions import StripedNotFoundException

def debug(msg):
    open("/tmp/striped.log","a").write(
            "%s: %s\n" % (time.ctime(time.time()), msg))

class   StripedConfiguration:
    def __init__(self, path):
                
        self.Config = yaml.load(open(path, "r").read())
        
        server_cfg = self.Config.get("Server", {})
        self.Buckets = sorted(server_cfg["buckets"])
        self.OperationTimeout = float(server_cfg.get("operation_timeout", 10.0))
        self.DataTTL = server_cfg.get("DataTTL", 30*24*3600)
        self.MetadataTTL = server_cfg.get("MetadataTTL", 24*3600)
        self.TagsConnect = self.Config.get("Tags", {}).get("connect")
        self.AAInfo = self.Config.get("Authorizations")     # { user: { "password": <password>,
                                                            #            "actions": [action, action...]}
    def getPassword(self, role, user):
        user_info = self.AAInfo.get(user)
        #print "user_info: %s %s %s" % (action, user, user_info)
        if user_info is None:   return None
        roles = user_info.get("roles", [])
        #print "roles:", roles
        if role in roles or "*" in roles:
            #print "password:", user_info.get("password")
            return user_info.get("password")
        else:
            return None
            
def cacheability_control(method):
    def decorated(*params, **args):
        resp = method(*params, **args)
        if args.get("cache","yes") != "yes":
            resp.headers["Cache-control"] = "no-store"
        return resp
    return decorated
            
class StripedHandler(WebPieHandler):

    def __init__(self, request, app, path=None):
        WebPieHandler.__init__(self, request, app, path)
        self.App.init(request)
        self.Config = self.App.Config
        self.MetadataTTL = self.Config.MetadataTTL
        self.DataTTL = self.Config.DataTTL

    def hello(self, req, relpath, **args):
        return Response("hello")

    def index(self, req, relpath, **args):
        return self.render_to_response("index.html")

    def authenticate(self, req, role):
        body = req.body
        #print folder.Name
        ok, header = digest_server(role, req.environ, self.App.authorizeUser)
        if not ok:
            resp = Response("Authorization required", status=401)
            if header:
                #print "Header: %s %s" % (type(header), header)
                resp.headers['WWW-Authenticate'] = header
            return False, resp
        user = header
        return True, user

    def datasets(self, req, relpath, **args):
        ds_last = {}
        buckets = self.Config.Buckets
        ds_info = {}
        ds_to_bucket = self.App.scanDatasets()
        for ds in self.App.datasets():
            cb = self.App.backendForDataset(ds)
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
        
    def rescan(self, req, relpath, **args):
        self.App.scanDatasets()
        
    def dataset_info(self, req, relpath, ds=None, **args):   
        self.App.scanDatasets()
        backend = self.App.backendForDataset(ds)
        if backend is None:    
            return Response("Dataset %s is not found" % (ds,), status=404)
        try:    schema = backend.schema(ds)
        except StripedNotFoundException:
            return Response("Dataset %s is not found" % (ds,), status=404)
        ne = 0
        ng = 0
        files = set()
        for rginfo in backend.RGInfos(ds):
            ne += rginfo["NEvents"]
            ng += 1
            for s in rginfo["Segments"]:
                files.add(s["FileName"])
        branches = [ (bname, sorted(bdict.items())) for bname, bdict in schema["branches"].items() ]
        return self.render_to_response("dataset_info.html", ds=ds, nfiles = len(files), nevents = ne, ngroups = ng,
                attributes = schema["attributes"].items(),
                branches = branches,
                )
        
    def ____buckets(self, req, relpath, **args):
        #print self.App.bucketInfoURL
        data = urllib2.urlopen(self.App.bucketInfoURL).read()
        data = json.loads(data)
        data_lst = []
        my_buckets = self.Config.Buckets
        for binfo in data:
            bname = binfo["name"]
            if bname in my_buckets:
                basic_stats = binfo["basicStats"]
                memUsed = basic_stats["memUsed"]
                diskUsed = basic_stats["diskUsed"]
                items = basic_stats["itemCount"]
                ops = basic_stats["opsPerSec"]
                quotaUsed = basic_stats["quotaPercentUsed"]
                datasets = [ds for ds in self.App.datasets() 
                                if self.App.bucketForDataset(ds) == bname]
                data_lst.append((bname, items, float(diskUsed)/1024/1024/1024, 
                                    float(memUsed)/1024/1024/1024, quotaUsed, ops, 
                                    sorted(datasets)))
        data_lst.sort()     # by bname
        return self.render_to_response("buckets.html", data = data_lst)

    #
    # Data
    #

            
    def dataset_list(self, req, relpath, **args):
        self.App.scanDatasets()
        lst = sorted(self.App.datasets())
        return Response(json.dumps(lst), content_type="text/json")
        
    def dataset_schema(self, req, relpath, ds=None, **args):
        backend = self.App.backendForDataset(ds)
        if not backend:
            return Response("Dataset %s is not found" % (ds,), status=404)
        try:    
            schema = backend.schema(ds)
        except StripedNotFoundException:
            return Response("Dataset %s is not found" % (ds,), status=404)
        return Response(json.dumps(schema), content_type="text/json")
        
    @cacheability_control
    def stripe(self, req, relpath, ds=None, rgid=None, column=None, compressed="no", **args):
        #
        # stripe?ds=<dataset>&rgid=<rgid>&column=<column>
        #
        compressed = compressed == "yes"
        backend = self.App.backendForDataset(ds)
        key = stripe_key(ds, column, rgid)
        data = backend.get_data([key]).get(key)
        if data is None:
            return Response("Stripe %s %s %s not found" % (ds, column, rgid),
                    status=400)
        if compressed:
            data = zlib.compress(data, 1)   
        return Response(data, cache_expires=self.DataTTL)
        
    @cacheability_control
    def data(self, req, relpath, ds=None, key=None, compressed="no", **args):
        as_json = args.get("json","no") == "yes"
        if ds is not None:
            ds = urllib.unquote(ds or "")
        compressed = compressed == "yes"
        backend = self.App.backendForDataset(ds)
        if backend is None:
            return Response("Can not find bucket", status=400)
        key = urllib.unquote(key)
        data = backend.getStandaloneData([key],  dataset_name=ds, as_json=as_json)[key] 
        if data is None:
            return Response("Not found", status=400)
        if as_json:
            data = json.dumps(data)
            return Response(data, cache_expires=self.DataTTL, content_type="text/json")
        if compressed:
            data = zlib.compress(data, 1)   
        return Response(data, cache_expires=self.DataTTL)
        
    def add_data(self, req, relpath, key=None, ds=None, **args):
        if req.environ.get("REQUEST_METHOD") != "POST":
            return Response("Bad request", status=402)

        ok, resp = self.authenticate(req, "add_data")
        if not ok:  return resp
        
        backend = self.App.backendForDataset(ds)
        backend.putStandaloneData({key:bytes(req.body)}, dataset_name=ds)
        return Response("OK")
        
    def token(self, req, relpath, role=None, ds=None, **args):
	#
	# Dataset (ds) is not use for now
	#
        if isinstance(role, unicode):
            role = role.encode("ascii", "ignore")
        if isinstance(ds, unicode):
            ds = ds.encode("ascii", "ignore")
        ok, resp = self.authenticate(req, role)
        #print "authenticate:", ok, type(resp), resp
        if not ok:  
            return resp
        user = resp
        token = self.App.tokenForUserRole(user, role)
        if not token:
            return Response("Unauthorized", status=403)
        return Response(token, content_type="text/plain")
        
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

    def upload_stripes(self, req, relpath, ds=None, rgid=None, **args):
        token = req.headers.get("X-Authorization-Token")
        if not token:
            return Response("Unauthorized", status=403)
        authorized, extra = self.App.authorizeToken(token, "upload_stripes")
        if not authorized:
            return Response("Unauthorized", status=403)
        payload = extra
        username = extra["identity"]
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
            #print type(cn), cn
            if val is not None:
                if compressed:
                    val = zlib.compress(val, 1)                
                data_out.append((cn, len(val), val))
        header = " ".join(("%s:%d" % (cn, n) for cn, n, v in data_out)) + "\n"
        
        def data_iterator(header, data):
            yield bytes(header)
            for cn, n, v in data:
                yield bytes(v)
                
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
                yield bytes(delimiter.join(buf))
                buf = []
            buf.append(l)
        if buf:
            yield bytes(delimiter.join(buf))

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
            self.Backend = CouchBaseBackend(self.BucketName, config = self.CBConfig)
            #try: self.Backend = CouchBaseBackend(self.BucketName, config = self.CBConfig)
            #except: return None
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
                #print "creating keeper for %s" % (bucket,)
                self.Backends[bucket] = BackendKeeper(self.CBConfig, bucket, self.StaleTimeout)
            backend = self.Backends[bucket].backend()
            #print "returning %s" % (backend,)
            return backend 
            
    __getitem__ = backend
            

def tidy_shape(shape):
    items = [x if x else "*" for x in shape]
    return "(" + ", ".join(["%s" % (x,) for x in items]) + ")"

class StripedApp(WebPieApp):

    TokenTTL = 3600*24
    TokenLeeway = 600

    def __init__(self, root_class):
        WebPieApp.__init__(self, root_class)

        self.Config = StripedConfiguration(os.environ['STRIPED_SERVER_CFG'])
        self.CouchBaseCfg = CouchBaseConfig(envVar = 'COUCHBASE_BACKEND_CFG')
        self.ConnectionPool = CBConnectionPool(self.CouchBaseCfg)
        self.scanDatasets()
        
        self.Secret = generate_secret(128)
        
        templates = os.environ['JINJA_TEMPLATES_LOCATION']
        self.initJinjaEnvironment(tempdirs = templates, 
            filters = {"tidy_shape": tidy_shape}, 
            globals = {})
            
    def authorizeUser(self, role, user):
	#print "authorizeUser(%s, %s)" % (role, user)
        return self.Config.getPassword(role, user)

    @app_synchronized
    def tokenForUserRole(self, username, role):
        token = SignedToken({"identity":username, "role":role}, self.TokenTTL).encode(self.Secret)
        return token

    @app_synchronized
    def authorizeToken(self, token, role):
        try:    token = SignedToken.decode(token, self.Secret, verify_times=True, leeway=self.TokenLeeway)
        except Exception as e:
            return False, str(e)
        payload = token.Payload
        if role == payload.get("role"):
            return True, payload
        else:
            return False, "Role not authorized"
        
    @property
    def defaultBucket(self):
        return self.CouchBaseCfg.DefaultBucket
     
    @property   
    def bucketInfoURL(self):
        return self.CouchBaseCfg.BucketInfoURL

    @app_synchronized        
    def init(self, request, force = False):
        pass
                    
    @app_synchronized
    def scanDatasets(self):
        #print "scan"
        dataset_map = {}        # dataset name -> bucket name
        for bucket in self.Config.Buckets:
            #print "bucket=", bucket
            backend = self.backendForBucket(bucket)
            for ds in backend.datasets():
                dataset_map[ds] = bucket
        self.DatasetMap = dataset_map
        return dataset_map
        
    def buckets(self):
        return self.Config.buckets()
        
    def datasets(self):
        #self.scanDatasets()
        return sorted(self.DatasetMap.keys())
        
    def bucketForDataset(self, dataset):
        if not dataset:
            return self.defaultBucket
        return self.DatasetMap.get(dataset)
        
    def backendForDataset(self, dataset):
        #self.scanDatasets()
        bucket_name = self.bucketForDataset(dataset)
        if not bucket_name: return None
        return self.backendForBucket(bucket_name)
        
    def backendForBucket(self, bucket_name):
        return self.ConnectionPool[bucket_name]
        
    def bucketURLForDataset(self, dataset):
        bucket = self.bucketForDataset(dataset)
        return self.Config.ServerURL + "/" + bucket + "?operation_timeout=%f" % (self.OperationTimeout,)
        
    def tagsDatabase(self):
        global TagsConnection
        if TagsConnection is None and self.Config.TagsConnect is not None:
            TagsConnection = psycopg2.connect(self.Config.TagsConnect)
        return TagsConnection
        
        
#print "creating application..."       
application = StripedApp(StripedHandler)

        
if __name__ == '__main__':
    import sys
    
    cfg = StripedConfiguration(None, sys.argv[1])
    for ds in sorted(cfg.datasets()):
        print ds, cfg.bucketForDataset(ds)
