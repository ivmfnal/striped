from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR, gethostbyaddr
import json, time, multiprocessing, os, traceback, sys

from striped.client import StripedClient
from striped.common import WorkerRegistryPinger, MyThread, DXMessage, WorkerRequest, DataExchangeSocket
from striped.pythreader import TaskQueue, Task, PyThread, synchronized

from SocketWorker2 import SocketWorkerBuffer 
#from sandbox import sandbox_import_module
from StripedWorker2 import WorkerDriver
from striped.common.exceptions import StripedNotFoundException
from striped.common.dataEncoder import decodeData, encodeData
from BulkStorage import BulkStorage
from striped.common import Tracer, BulkDataTransport


def distribute_items(lst, n):
    #print "distribute_items_simple(%d, %d)" % (len(lst), n)
    N = len(lst)
    k = N % n
    m = (N-k)/n
    i = 0
    out = []
    for _ in xrange(k):
        out.append(lst[i:i+m+1])
        i += m+1
    for _ in xrange(n-k):
        out.append(lst[i:i+m])
        i += m
    return out

class WorkerTask(Task):
    pass

    

class Worker(multiprocessing.Process):

    def __init__(self, wid, nworkers, striped_server_url, logfile, cache_limit, module_storage):
        multiprocessing.Process.__init__(self)
        self.ID = wid
        self.NWorkers = nworkers
        self.Client = StripedClient(striped_server_url, cache="long", cache_limit=cache_limit, log=self.log)
        self.ModuleStorage = module_storage
        self.Stop = False
        self.LogFile = None
        self.Sock = socket(AF_INET, SOCK_STREAM)
        self.Sock.bind(("127.0.0.1", 0))
        self.Port = self.Sock.getsockname()[1]
        self.Address = self.Sock.getsockname()
        self.Tasks = TaskQueue(2, capacity=10)
        if logfile != None:
            self.LogFile = open(logfile, "w")
            
    def log(self, msg):
        msg = "Worker %d port=%d %s: %s" % (self.ID, self.Port, time.ctime(time.time()), msg)
        print msg
        if self.LogFile is not None:
            self.LogFile.write(msg+"\n")
            self.LogFile.flush()
                    
    def run(self):
        signal.signal(signal.SIGINT, self.sigint)
        self.Sock.listen(10)
        while not self.Stop:
            self.log("----- accepting new connection...")
            sock, addr = self.Sock.accept()             # synchronously run only 1 job at a time, for now
            dxsock = DataExchangeSocket(sock)
            msg = dxsock.recv()
            if msg and msg.Type == "worker_task":
                worker_params = WorkerParams.fromDXMsg(msg)
                frames = json.loads(msg["frames"])
                wid = msg["wid"]
                try:
                    self.runWorker(worker_params, dxsock, frames, wid)
                except:
                    formatted = traceback.format_exc()
                    self.log("jid/wid=%s/%s: SocketWorkerServer.runWorker() exception:\n%s" % (worker_params.JID, wid, formatted,))
                    try:    
                        dxsock.send(DXMessage("exception").append(info=formatted))
                    except:
                        self.log("Error sending 'exception' message:\n%s" % (traceback.format_exc(),))
            self.log("----- closing socket")
            dxsock.close()
                    
    def runWorker(self, params, dxsock, frames, wid):
        buffer_id = "%s_%s" % (params.JID, self.ID)
        buffer = SocketWorkerBuffer(buffer_id, dxsock, params.HDescriptors, log=self.log)
        #worker_module = sandbox_import_module(module_name, ["Worker"])
        worker_module = __import__(params.WorkerModuleName, {}, {}, ["Worker"])

        T = Tracer()


        bulk_storage = None
        if params.BulkDataName:
                with T["open_bulk_storage"]:
                        bulk_storage = BulkStorage.open(params.BulkDataName)
                        #print "Worker: len(bulk_storage)=%d" % (len(bulk_storage),)
        
        worker_class = worker_module.Worker
        dataset_name = params.DatasetName
        user_params = params.UserParams
        use_data_cache = params.UseDataCache
        jid = params.JID

        data_mod_client = None
        if params.DataModURL is not None and params.DataModToken is not None:
            data_mod_client = StripedClient(params.DataModURL, data_modification_token=params.DataModToken)
            
        worker = WorkerDriver(jid, wid, self.Client, worker_class, dataset_name, frames, self.NWorkers, buffer, 
                user_params, bulk_storage, use_data_cache, 
                data_mod_client,
                tracer = T, log = self.log)
        self.log("Worker driver created for frames: %s" % (frames,))
        with T["worker.run"]:
            nevents = worker.run()
        self.log("jid/wid=%s/%s: worker.run() ended with nevents=%s" % (jid,wid,nevents))
        
        buffer.close(nevents)
        del sys.modules[params.WorkerModuleName]
        self.log("\n"+T.formatStats())
        self.log("jid/wid=%s/%s: exit from runWorker" % (jid,wid))

    def sigint(self, signum, frame):
        self.Stop = True
        sys.exit(1)
        
        
        
class WorkerParams(object):
    def __init__(self, jid, data_server_url, dataset_name,
                    hdesriptors, 
                    module_name, user_params, bulk_data_name,
                    use_data_cache, data_mod_url, data_mod_token):
        #
        # hdescriptors: {hist_name: hist_descriptor}
        #
        self.JID = jid
        self.DataServerURL = data_server_url         # not used for now, but worth keeping
        self.DatasetName = dataset_name
        self.HDescriptors = hdesriptors
        self.WorkerModuleName = module_name
        self.UserParams = user_params
        self.UseDataCache = use_data_cache
        self.DataModURL = data_mod_url
        self.DataModToken = data_mod_token
        self.BulkDataName = bulk_data_name
        
    def toDXMsg(self):
        msg = DXMessage("worker_task", jid=self.JID,
            worker_module_name = self.WorkerModuleName, 
            bulk_data_name = self.BulkDataName,
            dataset_name = self.DatasetName,
            use_data_cache = "yes" if self.UseDataCache else "no",
            data_server_url = self.DataServerURL,
            data_mod_url = self.DataModURL,
            data_mod_token = self.DataModToken)
        msg.append(
            histograms = json.dumps(self.HDescriptors),
            user_params = self.UserParams           # this is encoded by the job server, do not decode it yet
        )
        return msg
        
    @staticmethod
    def fromRequest(request, module_name):
        #print "fromRequest: UserParams=%s, <%s>" % (type(request.UserParams), request.UserParams)
        #print "WorkerParams.fromRequest: bulk_data_name=", bulk_data_name
        wp = WorkerParams(request.JID, request.DataServerURL, request.DatasetName,
            request.HDescriptors, module_name, request.UserParams, request.BulkDataName,
            request.UseDataCache, request.DataModURL,
            request.DataModToken)
        return wp
        
    @staticmethod
    def fromDXMsg(msg):
        assert msg.Type == "worker_task"
        jid = msg["jid"]
        dataset_name = msg["dataset_name"]
        worker_module_name = msg["worker_module_name"]
        data_server_url = msg["data_server_url"]
        histograms = json.loads(msg["histograms"])
        #print "fromDXMsg: histograms:", histograms
        user_params = decodeData(msg["user_params"])    # this comes pickled straight from the client, the job server passes it without decoding
        use_data_cache  = msg.get("use_data_cache", "yes") != "no"
        data_mod_token = msg.get("data_mod_token")
        data_mod_url = msg.get("data_mod_url")
        bulk_data_name = msg.get("bulk_data_name")
        #print "WorkerParams: fromDXMsg: bulk_data_name=", bulk_data_name
        params = WorkerParams(jid, data_server_url, dataset_name, histograms, worker_module_name, 
            user_params, bulk_data_name,
                use_data_cache, data_mod_url, data_mod_token)
        return params
        
        
class WorkerInterface(PyThread):

    def __init__(self, accumulator, worker_address, params, wid, frames):
        PyThread.__init__(self)
        self.WID = wid
        self.Accumulator = accumulator
        self.WorkerAddress = worker_address
        self.Params = params
        self.Frames = frames
        self.Done = False
        
    def run(self):
        wsock = DataExchangeSocket.connect(self.WorkerAddress)
        msg = self.Params.toDXMsg()
        msg.append(frames = json.dumps(self.Frames))
        msg.append(wid = self.WID)
        #print msg.Body
        wsock.send(msg)
        done = False
        while not done:
            msg = wsock.recv()
            if not msg or msg.Type == "end":
                done = True
            else:
                self.Accumulator.messageFromWorker(self, msg)
            
class AccumulatorDriver(Task):

    class JobInterface(object):
        def __init__(self, driver):
            self.Driver = driver

        @property
        def job_id(self):
            return self.Driver.JID
            
        def message(self, text):
            self.Driver.message(text)
            
    class DBInterface(object):
        #
        # dummy for now
        #
        def __init__(self, driver):
            self.Driver = driver
        
        
    def __init__(self, dxsock, request, workers, storage, bulk_data_transport):
        Task.__init__(self)
        self.DXSock = dxsock
        self.Request = request
        self.JID = request.JID
        self.Workers = workers
        self.ModuleStorage = storage
        self.Accumulator = None
        self.EventsAccumulated = 0
        self.EventsLastDataForward = 0
        self.T = Tracer()
        self.BulkDataTransport = bulk_data_transport

    def log(self, msg):
        print ("AccumulatorDriver(%s): %s" % (self.JID, msg))
        
    def run(self):
        try:
            storage = None
            bulk_data = None
            
            frames = self.Request.RGIDs
            frames_by_worker = distribute_items(frames, len(self.Workers))
            worker_module_name = "m_%s_%s" % (os.getpid(), self.Request.JID)     
            module_file = "%s/%s.py" % (self.ModuleStorage, worker_module_name)
            open(module_file, "w").write(self.Request.WorkerText)

            params = WorkerParams.fromRequest(self.Request, worker_module_name)

            #
            # Store bulk data in shared memory
            #
            if self.Request.BulkDataName:
                with self.T["wait_for_bulk_data"]:
                    t0 = time.time()
                    bulk_data = self.BulkDataTransport.pop(self.Request.BulkDataName, timeout=30)
                    t1 = time.time()
                    self.log("bulk data %s received, %d bytes encoded, %.2f wait time" % (self.Request.BulkDataName, len(bulk_data), t1-t0))
                    bulk_data = decodeData(bulk_data)
                with self.T["store_bulk_data"]:
                    assert isinstance(bulk_data, dict)
                    n = sum([len(v.data)+1000 for v in bulk_data.values()])
                    n = int(n*1.1)+1000000      # for safety
                    storage = BulkStorage.create(params.BulkDataName, bulk_data)
                    storage.save()
                self.log("bulk data stored. %f MB allocated" % (float(n)/1024/1024,))

            #
            # Create Accumulator if specified
            #
            worker_module = __import__(worker_module_name, {}, {}, ["Accumulator"])
            if hasattr(worker_module, "Accumulator"):
                job_interface = self.JobInterface(self)
                db_interface = self.DBInterface(self)
                self.Accumulator = worker_module.Accumulator(
                    params.UserParams, bulk_data,
                    job_interface, db_interface
                )        

            worker_interfaces = []
            for iw, (w, frames) in enumerate(zip(self.Workers, frames_by_worker)):
                if frames:
                    wid = "%s/%d" % (self.Request.WID, iw)
                    wi = WorkerInterface(self, w.Address, params, wid, frames)
                    wi.start()
                    worker_interfaces.append(wi)

            for wi in worker_interfaces:
                wi.join()
            self.log("all worker interfaces closed")

            if self.Accumulator is not None:
                nevents = self.EventsAccumulated
                data = self.Accumulator.values()
                if data is not None:
                    with self.T["send accumulated data"]:
                        self.DXSock.send(DXMessage("data", events_delta = 0, format="encode")(data=encodeData(data)))

            #self.DXSock.send(DXMessage("flush", nevents=self.EventsAccumulated))
                        
        except:
            self.DXSock.send(DXMessage("exception").append(info=traceback.format_exc()))

        finally:
            self.DXSock.close()
            self.log("socket closed")

            if storage:
                    storage.unlink()
                    self.log("bulk storage unlinked")

            os.unlink(module_file)
            if module_file.endswith(".py"):
                try:    os.unlink(module_file+"c")
                except OSError:
                    pass
                
            self.log("\n"+self.T.formatStats())

    @synchronized
    def message(self, message):
        self.DXSock.send(DXMessage("message", nevents=self.EventsAccumulated).append(message=message))

    @synchronized            
    def messageFromWorker(self, worker_interface, msg):
        # Can be message, hist, stream, flush, exception
        if msg.Type == "data":
            storage = BulkStorage.open(msg["storage"])
            #print "Accumulator.messageFromWorker(data): keys:", storage.keys()
            events_delta = msg["events_delta"]
            #print "messageFromWorker:events_delta=", events_delta
            data = storage.asDict()
            if self.Accumulator is None:
                msg = DXMessage("data", events_delta = events_delta, format="encode")(data=encodeData(data))
                self.DXSock.send(msg)
            else:
                    self.EventsAccumulated += events_delta
                    through = None
                    try:
                        with self.T["accumulate"]:
                            through = self.Accumulator.add(data)
                    except:
                        self.DXSock.send(DXMessage("exception").append(info=traceback.format_exc()))
                    if through is not None:
                        delta = self.EventsAccumulated - self.EventsLastDataForward
                        with self.T["send through data"]:
                            msg = DXMessage("data", events_delta = delta, format="encode")(data=encodeData(through))
                            self.DXSock.send(msg)       
                        self.EventsLastDataForward = self.EventsAccumulated
            storage.unlink()
        else:
            self.DXSock.send(msg)       
         
        
class WorkerMaster(PyThread):

    def __init__(self, config, bulk_data_transport):
        PyThread.__init__(self)
        self.StripedServerURL = config["ServerURL"]
        self.RegistryAddress = (config["Registry"]["host"], config["Registry"]["port"])
        self.NWorkers = config["NWorkers"]
        self.Tag = config.get("Tag", "default")
        self.CacheLimit = config.get("CacheLimit", 1.0) * 1.0e9
        self.NJobsRunning = config.get("RunningJobs", 2)
        self.QueueCapacity = config.get("JobQueueCapacity", 10)
        self.ModuleStorage = config.get("ModuleStorage", "/tmp/modules")
        if not self.ModuleStorage in sys.path:
            sys.path.insert(0, self.ModuleStorage)
        self.Sock = None
        self.Workers = []
        self.Stop = False
        self.Accumulators = TaskQueue(self.NJobsRunning, capacity=self.QueueCapacity)
        self.BulkDataTransport = bulk_data_transport
        
        self.LogFile = config.get("LogFile")
        if self.LogFile is not None:
            logfile = "%s.log" % (logfile, port)
            self.LogFile = open(logfile, "w")

        
    def log(self, msg):
        print ("Worker master: %s" % (msg,))
        if self.LogFile is not None:
            self.LogFile.write(msg+"\n")
            self.LogFile.flush()
        
        
    def run(self):
        signal.signal(signal.SIGINT, self.sigint)

        self.Sock = socket(AF_INET, SOCK_STREAM)
        self.Sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.Sock.bind(("", 0))
        self.Sock.listen(10)

        port = self.Sock.getsockname()[1]
        pinger = WorkerRegistryPinger(self.RegistryAddress, port, self.Tag)
        pinger.start()

        #
        # Start workers
        #
        
        self.Workers = [Worker(i, self.NWorkers, self.StripedServerURL, self.LogFile, self.CacheLimit, self.ModuleStorage) 
                for i in range(self.NWorkers)]
        for w in self.Workers:
            w.start()
        nrunning = self.NWorkers
        
        while not self.Stop:
            sock, addr = self.Sock.accept()
            dxsock = DataExchangeSocket(sock)
            close_sock = True
            #print "Client connected: %s" % (addr,)
            self.log("Client connected: %s" % (addr,))
            
            # read job description JSON
            #print "reading params..."
            
            try:    msg = dxsock.recv()
            except:
                self.log("Can not read initial message. Closing the connection. Error:\n%s" % 
                            (traceback.format_exc(),))
                msg = None
            if msg and msg.Type == 'request':
                try:
                    request = WorkerRequest.fromDXMsg(msg)
                    self.log("Request received:\n  jid/wid: %s/%s\n  dataset: %s\n  data_url: %s\n  frames: %s\n" % (
                                request.JID, request.WID, request.DatasetName, request.DataServerURL,
                                request.RGIDs)
                    )
                    signature, t, salt, alg = msg["worker_authenticator"].split(":")
                    #print "worker_authenticator:", (signature, t, salt, alg)
                    key = pinger.Key
                    verified, reason = request.verifySignature(key, signature, t, salt, alg)
                    if not verified:
                        self.log("Signature verification failed: %s" % (reason,))
                        dxsock.send(DXMessage("exception").append(info="Authentication failed: %s" % (reason,)))
                    else:
                        self.Accumulators << AccumulatorDriver(dxsock, request, self.Workers, self.ModuleStorage, self.BulkDataTransport)
                        close_sock = False
                except:
                    self.log("Error processing the request. Closing the connection\n%s" % (traceback.format_exc(),))

            if close_sock:
                dxsock.close()

                    
    def sigint(self, signum, frame):
        print "SIGINT received. Terminating..."
        self.terminate()

    @synchronized
    def terminate(self):
        self.Stop = True
        self.Accumulators.hold()
        for t in self.Accumulators.activeTasks():
            t.terminate()
        self.Accumulators.flush()
        self.Sock.close()
            
        
if __name__ == '__main__':
    import sys, os, signal, getopt, yaml
    
    Usage = """python worker_master.py -c <config.yaml>"""
    opts, args = getopt.getopt(sys.argv[1:], "h?c:")
    opts = dict(opts)
                
    if "-h" in opts or "-?" in opts:
        print
        print Usage
        print
        sys.exit(1)
    
    config = opts["-c"]
    config = yaml.load(open(config, "r").read())


    transport = BulkDataTransport(config.get("BulkTransportPort", 1234))
    transport.start()

    wm = WorkerMaster(config, transport)
    wm.run()
