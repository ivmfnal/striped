import json, zlib, time, random, sys, traceback

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

if PY3:
	from urllib.error import HTTPError, URLError
	from urllib.request import urlopen, Request
	import urllib.request, urllib.parse, urllib.error
	from urllib.parse import quote
else:
	from urllib2 import HTTPError, URLError, urlopen, Request
	from urllib2 import quote

import numpy as np
#from femtocode.definitons import Dataset, ColumnName, Column, Segment, Schema
#from typesystem import Schema

from striped.common import synchronized, Lockable, parse_data
from striped.common.exceptions import StripedNotFoundException
from .DataCache import DataCache

MAX_URL_LENGTH = 1000

class RGInfoSegment(object):

    def __init__(self, data):
        self.NEvents        = data["NEvents"]
        self.FileName       = data["FileName"]
        self.FileIndex      = data.get("FileIndex")
        self.BeginEvent     = data["BeginEvent"]

class RGInfo(object):

    def __init__(self, data):
        self._Version       = data["_Version"]
        self.NEvents        = data["NEvents"]
        self.RGID           = data["RGID"]
        self.BeginEventID   = data.get("BeginEventID")
        self.Segments       = [RGInfoSegment(s) for s in data["Segments"]]
        self.Profile        = data.get("Profile", {}) or {}
        self.Metadata       = self.Profile      # alias
        
    def meta(self, name):
        return self.Metadata.get(name)
        
    def __str__(self):
        return "RGInfo(RGID:%s, NEvenets:%s, BeginEventID:%s, Version:%s, segments:%s)" % (
                    self.RGID, self.NEvents, self.BeginEventID, self._Version, len(self.Segments))
            
class StripedColumnDescriptor(object):

    def __init__(self, desc_dict):
        #print type(desc_dict), desc_dict
        self.Type = desc_dict.get("type")
        self.NPType = desc_dict.get("type_np")
        self.ConvertToNPType = desc_dict.get("convert_to_np")
        self.Depth = desc_dict.get("depth")
        self.SizeColumn = desc_dict.get("size_column")
        self.ParentArray = desc_dict.get("parent_array")
        self.Shape = tuple(desc_dict.get("shape", ()))
        #print desc_dict
        
    def __str__(self):
        return "StripedColumnDescriptor(depth:%s, type:'%s', parent_array:'%s', size_column:'%s')" % \
            (self.Depth, self.Type, self.ParentArray, self.SizeColumn)

    @property
    def fixedShape(self):
        if not self.Shape:  return ()
        i = len(self.Shape) - 1
        while i >= 0:
            if self.Shape[i] is None:
                return tuple(self.Shape[i+1:])
            i -= 1
        return tuple(self.Shape)
        
    def asJSON(self):
        return json.dumps({
            "type":             self.Type,
            "type_np":          self.NPType,
            "convert_to_np":    self.ConvertToNPType,
            "depth":            self.Depth,
            "size_column":      self.SizeColumn,
            "shape":            self.Shape,
            "parent_array":     self.ParentArray})
        
    __repr__ = __str__
    
class SizeColumnDescriptor(object):

    def __init__(self):
        #print type(desc_dict), desc_dict
        self.Type = "int"
        self.NPType = "<i8"
        self.ConvertToNPType = "i8"
        self.Depth = 0
        self.SizeColumn = None
        self.ParentArray = None
        self.Shape = ()
        #print desc_dict
        
    @property
    def fixedShape(self):
        return ()

    def __str__(self):
        return "SizeColumnDescriptor()"
        
    __repr__ = __str__
    
class StripedColumn(object):

    def __init__(self, client, dataset, name, descriptor=None):
        self.Name = name
        self.Dataset = dataset
        self.DatasetName = dataset.Name
        self.UseMetaCache = client.UseMetaCache
        self.UseDataCache = client.UseDataCache
        if isinstance(descriptor, dict):    descriptor = StripedColumnDescriptor(descriptor)
        self.Descriptor = descriptor            # if preloaded
        if self.issize and self.Descriptor is None:
            self.Descriptor = SizeColumnDescriptor()
            
        self.Client = client
        
    def __str__(self):
        return "StripedColumn(%s)" % (self.Name,)
        
    __repr__ = __str__

    @staticmethod
    def isSizeName(column_name):
        return column_name.endswith(".@size")
    
    @property
    def issize(self):
        return StripedColumn.isSizeName(self.Name)

    isSize = issize
    
    @property
    def sizeColumn(self):
        #
        # returns StripedColumn object for physical size column, e.g.:
        #   Muon.pt -> Muon.@size
        #   Muon.pt.@size -> None   - menaingless
        #   EnvInfo.id -> None      - None for level 0 columns
        #
        if self.descriptor.SizeColumn:
            return self.Dataset.column(self.descriptor.SizeColumn)  
        else:
            return None         
        
    @property
    def descriptor(self):
        if self.issize:   return SizeColumnDescriptor()
        if self.Descriptor is None:
            data = self.Client.requestWithRetries("./column_desc?ds=%s&column=%s" %
                    (self.DatasetName, self.Name), bypass_cache=not self.UseMetaCache).read()
            self.Descriptor = StripedColumnDescriptor(json.loads(data))
        #print "column desc for %s:%s:" % (dataset, column), desc
        return self.Descriptor

    def _decode_list(self, size, data, depth):
        if depth == 0:
            return data, size, []
        if depth == 1:
            out = []
            i = 0
            for s in size:
                out.append(data[i:i+s])
                #print len(data[i:i+s].base)
                i += s
            return out, [], data[i:]
        else:
            out = []
            while size:
                n = size[0]
                segment, size, data = self._decode_list(size[1:], data, depth-1)
                out.append(segment)
            return out, size, data

    def assembleList(self, data, depth, size):
        #print "assembleList(data:%d, depth=%d, size:%d)" % (len(data), depth, len(size))
        if depth == 0:  return data
        out, size, data = self._decode_list(size, data, depth)
        #if len(size) or len(data):
        #    print ("leftover: size:%d data:%d" % (len(size), len(data)))
        assert len(size) == 0 and len(data) == 0        # make sure there are no leftovers
        return out
            
    def _____stripe(self, rgid, compress=False, assembled = False):
        columns = [self]
        if assembled:
            sc = self.sizeColumn
            if sc is not None:
                columns.append(sc)
        data = self.Client.stripes(self.DatasetName, columns, rgid, compress=compress)
        coldata = data[self.Name]
        if not assembled or sc is None:
            return coldata
        return self.assembleList(coldata, 1,            # FIXIT !!
                data[sc.Name])
        
        
    def stripeSizeArray(self, rgid, compress=False):
        c = self.sizeColumn
        if c is None:   return None
        return c.stripe(rgid, compress=compress)

class StripedDataset(Lockable):
    
    def __init__(self, client, name, preload_columns = []):
        self.Client = client
        self.UseMetaCache = client.UseMetaCache
        self.UseDataCache = client.UseDataCache
        self.Name = name
        self.Schema = None
        self.RGIDs = None
        self.ColumnNames = None
        self.ColumnsCache = client.ColumnsCache           # {column name -> column object} cache
        try:
            self.columns(preload_columns, include_size_columns=True)    # this will preload columns
        except StripedNotFoundException:
            raise StripedNotFoundException("Not all requested columns found in the dataset")

    @property
    def exists(self):
        try:    self.schema()
        except StripedNotFoundException:
            return False
        else:
            return True
   
    def schema(self, use_cache=None):
        if use_cache is None: use_cache = self.UseMetaCache
	#print "use_cache:", use_cache
        if not use_cache or not self.Schema:
            url = "./dataset_schema?ds=%s" % (self.Name, )
            schema = self.Client.requestWithRetries(url, bypass_cache=not use_cache).read()
            self.Schema = json.loads(schema)
        return self.Schema
    
    @property    
    def rgids(self, use_cache=None):
	#print "dataset.rgids: calling sever %s" % (self.Client.URLHead,)
        if use_cache is None: use_cache = self.UseDataCache
        if self.RGIDs is None or not use_cache:
            rgids = self.Client.requestWithRetries("./rgids?ds=%s" % (self.Name,), bypass_cache=not use_cache).read()
            self.RGIDs = json.loads(rgids)
        return self.RGIDs

    RGINFO_CHUNK_SIZE = 100
    
    def makeRanges(self, lst):
        # convert list of integers into list of tuples (i0,i1)
        
        if len(lst) == 0: return []
        
        lst = sorted(lst)
        i0 = lst[0]
        i1 = i0+1
        ranges = []
        for i in lst:
            if i > i1:
                # end of range
                ranges.append((i0, i1-1))
                i0, i1 = i, i+1
            else:
                i1 = i + 1
        ranges.append((i0, i1-1))
        return ranges
                
    def rginfo(self, rgids, use_cache=None):
        if use_cache is None:   use_cache = self.UseMetaCache

        def get_chunk(ranges):
            text = ",".join(["%d" % (r0,) if r1 == r0 else "%d:%d" % (r0, r1) for r0, r1 in ranges])
            url = "./rginfo?ds=%s&rgids=%s" % (self.Name, text)
            #sys.stderr.write("url=%s\n" %(url,))
            data = self.Client.requestWithRetries(url, bypass_cache=not use_cache).read()
            lst = json.loads(data)
            #print type(data[names[0]]), data[names[0]]
            return lst

        return_list = isinstance(rgids, (list, tuple))
        if not isinstance(rgids, (list, tuple)): rgids = [rgids]
        out = []
        ranges = self.makeRanges(rgids)
        #print "ranges:", ranges
        for i in range(0, len(ranges), self.RGINFO_CHUNK_SIZE):
            out += get_chunk(ranges[i:i+self.RGINFO_CHUNK_SIZE])

        if not return_list:  out = out[0]
        return out
        
    def rginfos(self, rgids):
        if isinstance(rgids, int):
            rgids = [rgids]
        return list(map(RGInfo, self.rginfo(rgids)))
        
    def nevents(self):
        return sum([r.NEvents for r in self.rginfos(self.rgids)])
        
    def columnsAndSizes(self, column_names):
        # return complete list of all columns, which need to be retrieved to reconstruct given columns set
        columns_dict = self.columns(column_names, include_size_columns=True)
        return sorted(columns_dict.keys())
    
    @property    
    def columnNames(self, use_cache=None):
        if use_cache is None: use_cache = self.UseMetaCache
        if self.ColumnNames is None or not use_cache:
            names = self.Client.requestWithRetries("./columns?ds=%s" % (self.Name,), bypass_cache=not use_cache).read()
            self.ColumnNames = json.loads(names)
        return self.ColumnNames
        
    def column(self, name):
        if StripedColumn.isSizeName(name):
            return StripedColumn(self.Client, self, name)
        return self.ColumnsCache.get(self.Name, name) or self.columns([name])[name]
        
    def sizeColumnFor(self, c):
        if c.descriptor.SizeColumn:
            return StripedColumn(self.Client, self, c.descriptor.SizeColumn)
        else:
            return None
    
    def columns(self, names, use_cache = None, include_size_columns = False):
        # names list can be long so that the resulting URL will exceed the URL size limit
        # split the list into smaller chunks and send them in separate requests

        if use_cache is None:   use_cache = self.UseMetaCache

        def get_chunk(names):
            url = "./column_descs?ds=%s&columns=%s" % (self.Name, ",".join(names))
            #sys.stderr.write("url=%s\n" %(url,))
            data = self.Client.requestWithRetries(url, bypass_cache=not use_cache).read()
            data = json.loads(data)
            #print type(data[names[0]]), data[names[0]]
            return dict(
                [(name, StripedColumn(self.Client, self, name, descriptor=desc)) for name, desc in data.items()] )
    
        names = sorted(names)       # help the web cache
        data_names = []
        out_dict = {}               # name -> StripedColumn object
        uncached = []
        for n in names:
            cached = self.ColumnsCache.get(self.Name, n)
            if cached is None:
                uncached.append(n)
            else:
                out_dict[n] = cached
        #sys.stderr.write("remaining columns: %s\n" % (purged_names,))
        names = uncached
        
        for n in names:
            if StripedColumn.isSizeName(n): # do not ask for size column descriptors
                out_dict[n] = StripedColumn(self.Client, self, n, SizeColumnDescriptor())
            else:
                data_names.append(n)

        chunk = []
        nchars = 0
        for name in data_names:
            if len(chunk)+nchars >= MAX_URL_LENGTH - 100:       # expected length of the comma-separated list
                out_dict.update(get_chunk(chunk))
                nchars = 0
                chunk=[]
            chunk.append(name)
            nchars += len(name)
            
        if chunk:
            out_dict.update(get_chunk(chunk))
            
        # check if any columns are missing in the dataset
        #print out_dict
        missing = [cn for cn in names if not cn in out_dict]
        if len(missing):
            raise KeyError("The following columns are not found in the dataset: %s" % (",".join(missing),))

        if include_size_columns:
            for cn, cc in out_dict.items():
                sc = cc.sizeColumn
                if sc is not None:  out_dict[sc.Name] = sc 

        for cn, cv in out_dict.items():
            self.ColumnsCache.put(self.Name, cn, cv)

        return out_dict

    def columnToBranch(self, column_names):
        columns_dict = self.columns(column_names, include_size_columns=True)
        return { cn: cc.descriptor.ParentArray for cn, cc in columns_dict.items() }

    @property    
    def allColumns(self):           # conveninence
        return self.columns(self.columnNames)

    def stripe(self, column, rgid, compress=False):
        return self.stripes([column], rgid, compress=compress)[column]
        
    def stripes(self, columns, rgid, compress=False, use_cache = None):
        if use_cache is None:   use_cache = self.UseDataCache
        #sys.stderr.write("stripes: %s\n" % (type(columns), ))
        if columns and isinstance(columns[0], str):
            columns_dict = self.columns(columns)
            columns = [columns_dict[cn] for cn in columns]
            #sys.stderr.write("columns converted: %s\n" % (columns,))
        return self.Client.stripes(self.Name, columns, rgid, compress=compress, use_cache = use_cache)
        
    def stripeSizes(self, columns, rgids, use_cache=None):
        #
        # returns dictionary:
        # { column_name: { rgid: size in bytes, ... }, ... }
        if use_cache is None:   use_cache = self.UseDataCache
        request_body = json.dumps({
            "ds":       self.Name,
            "columns":  [c.Name for c in columns],
            "rgids":    rgids
            })
        #column_names = ','.join([c.Name for c in columns])
        #rgids = ','.join("%d" % (rgid,) for rgid in rgids)
        data = self.Client.requestWithRetries("./stripes_sizes?ds=%s&columns=%s&rgids=%s" % (
                self.Name, 
                ",".join([c.Name for c in columns]), 
                ",".join(["%d" % (rgid,) for rgid in rgids])
            ),
            bypass_cache = not use_cache
        ).read()
        data = json.loads(data)
        return dict([(cn, dict(info)) for cn, info in data.items()])
        
    def putStripes(self, rgid, data):
        # data is dictionary {column_name -> npy_array}
        header = []
        body = []
        for cn, arr in data.items():
            cd = self.column(cn).descriptor
            data = bytes(np.asarray(arr, cd.NPType).data)
            body.append(data)
            header.append("%s:%s" % (cn, len(data)))
        body = " ".join(header) + "\n" + "".join(body)
        self.Client.requestWithRetries("./upload_stripes?ds=%s&rgid=%s" % (self.Name, rgid), body=body, 
            headers={"X-Authorization-Token":self.Client.dataModToken()})
        
    def taggedEvents(self, first_event_id, nevents, conditions):
        url = "./tagged_events?ds=%s&event_id_range=%d:%d&conditions=%s" % (
            self.Name, first_event_id, first_event_id+nevents,
            ",".join(conditions))
        data = self.Client.requestWithRetries(url).read()
        data = json.loads(data)
        return data
        
class ColumnDescCache(Lockable):

    Limit = 500
    LowWater = 400

    def __init__(self):
        Lockable.__init__(self)
        self.Cache = {}         # {"dataset:column" -> descriptor}
    
    def key(self, dataset, cname):
        return "%s:%s" % (dataset, cname)
    
    @synchronized
    def get(self, dataset, cname):
        return self.Cache.get(self.key(dataset, cname))
        
    @synchronized
    def put(self, dataset, cname, cdesc):
        self.Cache[self.key(dataset, cname)] = cdesc
        if len(self.Cache) > self.Limit:
            remove = random.sample(self.Cache.keys(), self.Limit-self.LowWater)
            for k in remove:
                del self.Cache[k]


class StripedClient(Lockable):

    def __init__(self, url_head, trace = None, flush_cache = False, cache="long",       # cache can be "long", "short" or "none"
                use_data_cache = True,  use_metadata_cache = True,
                cache_limit=1000**3,    # 1 GB
                data_modification_username = None, data_modification_password = None,
                data_modification_token = None,
                log=None):
        Lockable.__init__(self)
        self.URLHead = url_head
        self.ColumnDescriptors = {}
        self.CacheFlush = "_=%s" % (time.time(),) if flush_cache else ""
        if cache in ("short", "long"):
            self.StripeCache = DataCache(cache_limit)
        else:
            self.StripeCache = None
        self.Log = log
        self.ColumnsCache = ColumnDescCache()
        self.UseDataCache = use_data_cache
        self.UseMetaCache = use_metadata_cache

        self.DataModToken = data_modification_token
        self.DataModUsername = data_modification_username
        self.DataModPassword = data_modification_password
        self.DataModTokenBox = None

        if self.DataModUsername is not None and self.DataModPassword is not None:
            self.DataModTokenBox = TokenBox("%s/token?role=upload_stripes" % (self.URLHead,),
                            self.DataModUsername, self.DataModPassword)


	#print "Client: self.UseMetaCache=", self.UseMetaCache
        
    def useCache(self, **args):
        if "data" in args:
            self.UseDataCache = args["data"]
        if "metadta" in args:
            self.UseMetaCache = args["metadata"]
        
    @synchronized
    def log(self, msg):
        if self.Log is not None:
            self.Log("StripedClient: %s" % (msg,))
            
    @synchronized
    def dataModToken(self):
        token = self.DataModToken
        if token is None:
            if self.DataModTokenBox is None:
                raise ValueError("Data modifications not authorized")
            token = self.DataModTokenBox.token
        return token
        
    def requestWithRetries(self, url, headers={}, bypass_cache=False, timeout=60, body = None):
        t0 = time.time()
        delay = 0.5
        done = False

        if url.startswith("./"):    url = self.URLHead + url[1:]
        
        if self.CacheFlush:
            # append unique but meaninless query argument to make sure this client does not use some other client's web cache
            parts = url.split("?", 1)
            if len(parts) < 2:  parts.append("")
            head, query = tuple(parts)
            if not query:
                query = "?" + self.CacheFlush
            else:
                query += "&" + self.CacheFlush
            url = head + "?" + query

        if bypass_cache:
                if not '?' in url:
                        url += "?"
                else:
                        url += '&'
                url += "cache=no&_=%s" % (time.time(),)

        #print "requestWithRetries: url:", url

        while not done and time.time() < t0 + timeout:
            #sys.stderr.write("urllib2.urlopen(%s)\n" % (url,))
            try:
                #self.log("urlopen(%s)..." % (url,))
                request = Request(url, body, headers)
                response = urlopen(request, timeout=timeout/2)
                code = response.getcode()
                #self.log("success, http code=%s" % (code,))
            except HTTPError as error:
                self.log("HTTPError %s" % (error,))
                code = error.code
                response = None
                #sys.stderr.write("url: %s code:%s error:%s\n" % (url, code, error))
            except URLError as error:
                self.log("URLError %s" % (error,))
                code = 500
                response = None
            if code/100 == 2:   done = True
            elif code/100 == 4: raise StripedNotFoundException("'Not found' response for URL: %s" % (url,))
            elif code/100 == 5:
                # retry later
                tsleep = random.random() * delay
                sys.stderr.write("url:%s error:%s. will retry in %.3f seconds\n" % (url, error, tsleep))
                self.log("Error code: %s, will retry in %s seconds" % (code, tsleep))
                time.sleep(tsleep)
            else:
                raise ValueError("HTTP error %s" % (code,))
            delay *= 1.1
        if not done:
            self.log("giving up")
            raise ValueError("HTTP retry timeout")
        return response

    def datasets(self, use_cache=None):
        if use_cache is None:   use_cache = self.UseMetaCache
        #eturns dataset names as a list of strings
        response = self.requestWithRetries("./dataset_list", bypass_cache=not use_cache)
        return [StripedDataset(self, ds) for ds in json.loads(response.read())]
        
    def dataset(self, name, preload_columns = []):
        return StripedDataset(self, name, preload_columns=preload_columns)
        
    def raw_stripes(self, dataset, columns, rgid, use_cache=None, compress=False):
        if use_cache is None:   use_cache = self.UseDataCache
        out_data = {}

        column_list = ",".join(columns)
        url = "./stripes?ds=%s&columns=%s&rgid=%s&compressed=%s" % \
            (dataset, column_list, rgid, "yes" if compress else "no")
        #print "url:", url
        request = self.requestWithRetries(url, bypass_cache=not use_cache, timeout=120)
        data = request.read()
        #print "data: %s" % (repr(data[:100]),)
        header_end = data.index("\n")
        header = data[:header_end]
        i = header_end + 1
        #print "header: [%s]" %(header,)
        for w in header.split():
            try:    cn, length = w.split(":")
            except ValueError:
                sys.stderr.write("Error parsing header [%s] url:%s status:%s" % 
                        (header, url, request.getcode()))
                sys.stderr.write("request status=%s" % (request.getcode(),))
                sys.stderr.write(traceback.format_exc()+"\n")
                sys.exit(1)
            length = int(length)
            segment = data[i:i+length]
            if compress:    
                    segment=zlib.decompress(segment)
            out_data[cn] = segment
            i += length
        return out_data
            
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

    def convertSizeStripe(self, data):
        return np.asarray(np.frombuffer(data, "<i8"), "i8")

    def stripes(self, dataset, columns, rgid, compress=False, use_cache=None):
        # columns are StripedColumn objects
        #sys.stderr.write("client.stripes: columns=%s\n" % (columns,))
        
        if use_cache is None:   use_cache = self.UseDataCache
        
        if isinstance(dataset, StripedDataset):
            dataset = dataset.Name
        out_data = {}
        all_columns = sorted([c.Name for c in columns])
        missing_columns = all_columns
        
        
        if use_cache and self.StripeCache is not None:
            missing_columns = []
            for c in columns:
                cn = c.Name
                data = self.StripeCache.get((dataset, rgid, cn))
                if data is None:
                    if not cn in missing_columns:
                        missing_columns.append(cn)
                        #print "Client: column is not in cache:", cn
                        break
                else:
                    out_data[cn] = data

        #
        # Always ask for all columns even if only one is missing, to help web cache
        #
        if missing_columns: missing_columns = all_columns
        #print "Client: missing columns: ", missing_columns
        missing_data = {}
        if missing_columns:
            #missing_columns.sort()      # help HTTP cache
            missing_data = self.raw_stripes(dataset, missing_columns, rgid, compress=compress, use_cache=use_cache)
            #print "received columns: ", missing_data.keys()
                
            for c in columns:
                cn = c.Name
                if not cn in out_data:    
                    desc = c.descriptor
                    data = self.convertStripe(missing_data[cn], desc)
                    self.StripeCache.store((dataset, rgid, cn), data)
                    out_data[cn] = data
        return out_data
        
    def standaloneData(self, key, dataset=None, compress=False, use_cache=None, as_json=False):
        if use_cache is None:   use_cache = self.UseDataCache
        data = None
        if use_cache and self.StripeCache is not None:
            cache_key = (dataset, "@@data", key)
            #print "cache_key:", cache_key
            data = self.StripeCache.get(cache_key)
            #print "data cached:", data is not None
        if data is None:
            # assume binary for now
            url = "./data?key=%s&json=no&compress=%s" % \
                (quote(key), "yes" if compress else "no")
            if dataset:
                url += "&ds=%s" % (quote(dataset),)
            #print "url:", url
            request = self.requestWithRetries(url, bypass_cache=not use_cache)
            data = request.read()
            #print "data:", data
            data = parse_data(data)
            if as_json:
                data = json.loads(data)
            if use_cache and self.StripeCache is not None:
                self.StripeCache.store(cache_key, data)
        return data
        
if __name__ == '__main__':
    import sys, time, random
    
    def dataset_list():
    
        Usage="""
            python SrtripedClient.py <URLHead>
        """
        dataset = sys.argv[1]
        client = StripedClient(sys.argv[1])
        for ds in client.datasets():
            print(ds.Name)
    
    def rgids():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset>
        """
        client = StripedClient(sys.argv[1])
        dataset = sys.argv[2]
        rgids = client.dataset(dataset).rgids
        print (rgids)
    
    def column_names():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset>
        """
        client = StripedClient(sys.argv[1])
        dataset = client.dataset(sys.argv[2])
        
        for cn in dataset.columnNames:
            print (cn)
    
    def column_descs():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset>
        """
        client = StripedClient(sys.argv[1])
        dataset = client.dataset(sys.argv[2])
        all_columns = dataset.allColumns        # this is a dictionary { column name -> column object }
        for cn, c in all_columns.items():
            print("%s %s" % (cn, c.Descriptor))
    
    def rginfo():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset> <rgid0> <rgid1>
        """
        client = StripedClient(sys.argv[1])
        dataset = sys.argv[2]
        rgids = list(range(int(sys.argv[3]), int(sys.argv[4])+1))
        dataset = client.dataset(dataset)
        rginfos = dataset.rginfos(rgids)
        for rginfo in rginfos:
            print (rginfo)
            
    def column_desc():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset> <column>
        """
        client = StripedClient(sys.argv[1])
        dataset = sys.argv[2]
        dataset = client.dataset(dataset)
        column = dataset.column(sys.argv[3])
        print(column.descriptor)
        
    def multi_column_desc():
        Usage="""
            python SrtripedClient.py <URLHead> <dataset> <column>
        """
        client = StripedClient(sys.argv[1])
        dataset = sys.argv[2]
        dataset = client.dataset(dataset)
        column = dataset.column(sys.argv[3])
        print(column.descriptor)
            
    def stripes():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <column>,... <rgid>
        """
        client = StripedClient(sys.argv[1])
        ds = client.dataset(sys.argv[2])
        columns = sys.argv[3].split(",")
        columns = [ds.column(cn) for cn in columns]
        rgid = int(sys.argv[4])
        t0 = time.time()
        data = ds.stripes(columns, rgid)
        t1 = time.time()
        delta_t = t1-t0
        for cn, stripe in data.items():
            print("%s %s %s" % (cn, stripe[0].dtype, len(stripe)))
        
    def stripes_sizes():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <column>,...
        """
        client = StripedClient(sys.argv[1])
        dataset = client.dataset(sys.argv[2])
        columns = dataset.columns(sys.argv[3].split(",")).values()
        rgids = list(range(99))
        t0 = time.time()
        sizes_dict = dataset.stripeSizes(columns, rgids)
        print("Time: %s" % (time.time() - t0,))
        print (sizes_dict)
         
    def get_stripe():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <column> <rgid>
        """
        client = StripedClient(sys.argv[1])
        ds = client.dataset(sys.argv[2])
        column = ds.column(sys.argv[3])
        rgid = int(sys.argv[4])
        t0 = time.time()
        stripe = column.stripe(rgid)
        t1 = time.time()
        delta_t = t1-t0
        print("Stripe received. Time=%f, length=%d, type=%s" % (delta_t, len(stripe), stripe[0].dtype))
        print (stripe)
        
    def get_stripe_assembled():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <column> <rgid>
        """
        client = StripedClient(sys.argv[1])
        ds = client.dataset(sys.argv[2])
        column = ds.column(sys.argv[3])
        rgid = int(sys.argv[4])
        t0 = time.time()
        stripe = column.stripe(rgid, assembled=True)
        for x in stripe:
            print (x)
        
            
    def schema():
        Usage="""
            python StripedClient.py <URLHead> <dataset>
        """
        client = StripedClient(sys.argv[1])
        ds = client.dataset(sys.argv[2])
        schema = ds.schema
        print(son.dumps(schema, sort_keys=True, indent = 4, separators=(',',': ')))
        
    def tagged_events():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <event_id_min> <n events> <condition> ...
        """
        client = StripedClient(sys.argv[1])
        ds = client.dataset(sys.argv[2])
        eid_min = int(sys.argv[3])
        nevents = int(sys.argv[4])
        conditions = sys.argv[5:]
        for eid in ds.taggedEvents(eid_min, nevents, conditions):
            print (eid)
        
    def get_data():
        Usage="""
            python StripedClient.py <URLHead> <dataset> <data key>
        """
        client = StripedClient(sys.argv[1])
        dataset_name = sys.argv[2]
        key = sys.argv[3]
        data = client.standalone_data(dataset_name, key)
        print(repr(data))

    get_data()
        

        
        
