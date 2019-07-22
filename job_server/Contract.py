import sys, os, time, uuid, cPickle, random, multiprocessing
import numpy as np
from striped.pythreader import PyThread, Primitive, synchronized
from threading import Event
import socket, traceback, random, time
from striped.common import DXMessage, WorkerRequest, DataExchangeSocket, BulkDataSender

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
                
def distribute_items_modulo(lst, n):
    if n == 0:
        return []
    lists = []
    for _ in xrange(n):
        lists.append([])
    for i in lst:
        lists[i%n].append(i)
    return lists

class ParamsSender(multiprocessing.Process):

        def __init__(self, msg, dxsock):
                multiprocessing.Process.__init__(self)
                self.Msg = msg
                self.Sock = dxsock

        def run(self):
                self.Sock.send(self.Msg)

class SocketWorkerInterface(PyThread):

    def __init__(self, contract, wid, nworkers, params, worker_address, worker_key, log):
        PyThread.__init__(self)
        self.WID = wid
        self.Contract = contract
        self.WorkerAddress = worker_address
        self.WorkerKey = worker_key
        self.Params = params
        self.TStart = None
        self.NEvents = 0
        self.LastNEvents = 0
        self.Buffer = ""
        self.Abort = False
        self.Log = log
        self.log("interface created with worker address %s" % (self.WorkerAddress,))
        self.Created = time.time()
        
    def log(self, msg):
        #print "WorkerInterface %d: %s" % (self.WID, msg)
        if self.Log:
            self.Log("WorkerInterface %d: %s" % (self.WID, msg))
        else:
                print("WorkerInterface %d: %s" % (self.WID, msg))

    def abort(self):
        self.Abort = True
        
    def run(self):
        try:
            self.Contract.waitForStart()
            tstart = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:    
                sock.connect(self.WorkerAddress)
            except: 
                print ("Error connecting to %s" % (self.WorkerAddress,))
                raise

            dxsock = DataExchangeSocket(sock)

            #print "SocketWorkerInterface: connected to %s" % (self.WorkerAddress,)

            #print "SocketWorkerInterface: params:"
            #for k, v in self.Params.__dict__.items():
            #    if not k.startswith("_"):
            #        print k, v
            
            msg = self.Params.toDXMsg()
            signature, t, salt, alg = self.Params.generateSignature(self.WorkerKey)
            msg["worker_authenticator"] = "%s:%s:%s:%s" % (signature, t, salt, alg)

            dxsock.send(msg)

            self.log("Worker parameters sent. Time since created=%f" % (time.time() - self.Created,))

            eof = False
            hists = {}
            streams = {}
            while not eof:
                    #print "%s: readCunk..." % (self.WID,)

                if self.Abort:
                    sock.close()
                    break

                #print "DXMessage.fromSocket..."            
                msg = dxsock.recv()
                if msg is None:
                    eof = True
                    self.log("EOF")

                else:
                    self.log("message(%s)" % (msg.Type,))
                    if msg.Type == 'flush':
                        nevents = int(msg["nevents"])
                        events_delta = nevents - self.LastNEvents
                        self.LastNEvents = nevents
                        #print "buffer %s: updateReceived..." % (self.WID,)
                        self.Contract.updateReceived(self, hists, streams, events_delta)
                        #print "buffer %s: updateReceived done"  % (self.WID,)
                        hists = {}
                        streams = {}

                    elif msg.Type == 'message':
                        message = msg["message"]
                        nevents = msg["nevents"]
                        self.Contract.messageReceived(self, nevents, message)

                    elif msg.Type == 'hist':
                        hists[msg["hid"]] = msg["dump"]

                    elif msg.Type == 'stream':
                        name = msg["name"]
                        #format = msg["format"]
                        #assert format == "pickle", "Unknown stream serialization format %s" % (format,)
                        streams[name] = msg["data"] # do not unpickle yet !

                    elif msg.Type == 'data':
                            self.Contract.dataReceived(self, msg["events_delta"], msg["data"])

                    elif msg.Type == 'exception':
                        #print "Contract: exception received"
                        self.Contract.exceptionReceived(self, msg["info"])

                    elif msg.Type == 'data_load_failure':
                        self.Contract.dataLoadFailureReceived(self, msg["rgid"])
        finally:
            self.Contract.workerExited(self, 0, time.time() - tstart)
            self.Contract = None        # break circular dependencies

class Contract(Primitive):

    def __init__(self, jid, data_server_url, bulk_transport_port, dataset, job_desc, workers, callback_delegate, log, T):
    
        #print "SocketGangContract: worker_text=%s" % (worker_text,)
    
        Primitive.__init__(self)
        
        #print "Contract: data_server_url=", data_server_url
        self.JID = jid       
        self.BulkTransportPort = bulk_transport_port
        self.WorkerInterfaces = {}
        self.Params = {}
        self.DoneWorkers = {}
        self.CallbackDelegate = callback_delegate
        self.WorkerText = job_desc.WorkerText
        self.Workers = workers
        self.EventsProcessed = 0
        self.Log = log
        self.T = T
        self.StartEvent = Event()
        self.UseDataCache = job_desc.UseDataCache
        self.TotalEvents = None
        self.SelectedEvents = None
        self.SelectedFrames = None
        self.BulkData = job_desc.BulkData
        self.BulkDataName = "job_%s.bulk" % (self.JID,) if self.BulkData is not None else None
        
        nworkers = len(workers)     # for now

        with self.T["Contract.__init__"]:
        
                hdescriptors = {hid:h if isinstance(h, dict) else h.descriptor()
                                    for hid, h in job_desc.HDescriptors.items()}
                #print hdescriptors
                
                # assign rgids

                dataset.UseDataCache = self.UseDataCache
                self.NRGs, rgid_dist = self.distributeWork(nworkers, len(self.Workers), dataset, job_desc.FrameSelector, job_desc.Fraction)
                #print "Contract: work distribution: nworkers=%d, frames=%d" % (nworkers, self.NRGs)
                #print "   ", rgid_dist
                self.NWorkers = min(nworkers, len(self.Workers))
                
                if self.NRGs > 0:
                    #print "rgids_dist:", rgids_dist

                    worker_module_name = "wm_%s" % (uuid.uuid1(),)

                    for iw in xrange(len(self.Workers)):
                        rgid_list = rgid_dist[iw]
                        #print iw, len(rgid_list)
                        if len(rgid_list):
                            with self.T["Contract.__init__/create_params"]:
                                    self.Params[iw] = WorkerRequest(self.JID, iw, data_server_url, dataset.Name, rgid_list, self.NWorkers, 
                                        worker_module_name, job_desc.WorkerText, job_desc.HDescriptors, job_desc.UserParams, job_desc.UseDataCache,
                                        job_desc.DataModificationURL, job_desc.DataModificationToken, self.BulkDataName
                                        )
                                
                self.log("Contract created")

    def distributeWork(self, nworkers_job, nworkers_available, dataset, frame_selector, fraction):
        rgids_initial = sorted(dataset.rgids[:])
        nworkers = min(nworkers_job, nworkers_available)
        workers = range(nworkers_available)
        if nworkers < nworkers_available:
                rstate = random.getstate()
                seed = hash(dataset.Name)
                random.seed(seed)
                workers = random.sample(workers, nworkers)
                random.setstate(rstate)

        initial_distribution = distribute_items(rgids_initial, nworkers)

        rgids = set(rgids_initial)
        with self.T["Contract.__init__/distribute/rginfos"]:
                rginfos = dataset.rginfos(list(rgids))
        self.TotalEvents = self.SelectedEvents = sum([i.NEvents for i in rginfos])
        rginfo_dict = {i.RGID:i for i in rginfos}
        if frame_selector is not None:
            rgids = set()
            for rginfo in rginfos:
                if frame_selector.eval(rginfo.Profile):
                    rgids.add(rginfo.RGID)

        if isinstance(fraction, float) and len(rgids):
            nrgids = float(len(rgids)) * fraction
            n = int(nrgids)
            if n < nrgids:
                n += 1
            if n < len(rgids):
                rstate = random.getstate()
                seed = hash(dataset.Name)
                random.seed(seed)
                rgids = set(random.sample(rgids, n))
                random.setstate(rstate)

        rgid_dist = [[][:] for _ in xrange(nworkers_available)]
        for iw, lst in enumerate(initial_distribution):
            iw_actual = workers[iw]
            for rgid in lst:
                if rgid in rgids:
                    rgid_dist[iw_actual].append(rgid)

        self.SelectedFrames = sorted(list(rgids))
        self.SelectedEvents = sum([rginfo_dict[rgid].NEvents for rgid in rgids])
        #self.log("RG distributed: %s" % (rgid_dist,))
        return len(rgids), rgid_dist

    def start(self):
        self.WorkerInterfaces = {}

        transport_client = None
        if self.BulkData is not None:
            addresses = sorted([wi.Addr[0] for wi in self.Workers])
            #print "Contract.start(): bulk transfer addresses: %s" % (addresses,)
            random.shuffle(addresses)
            transport_client = BulkDataSender(self.BulkDataName, self.BulkData, self.BulkTransportPort, addresses)
            transport_client.start()

        for i, params in self.Params.items():
            wi = self.Workers[i]
            w = SocketWorkerInterface(self, i, self.NWorkers, params, wi.Addr, wi.Key, self.Log)
            self.WorkerInterfaces[i] = w
            
        for w in self.WorkerInterfaces.values():
            with self.T["WorkerInterface.start()"]:
                w.start()
            #time.sleep(0.001)
        
        self.StartEvent.set()
        
    def waitForStart(self):
        self.StartEvent.wait()
            
    def log(self, msg):
        if self.Log:
            self.Log("Contract: %s" % (msg,))
        else:
                print("Contract: %s" % (msg,))
            
    @synchronized
    def nevents(self):
        n = sum(w.NEvents for w in self.WorkerInterfaces.values()) + sum(w.NEvents for w in self.DoneWorkers.values())
        #print "sum of workers:", n
        return n
    
    @synchronized
    def updateReceived(self, worker, hists, streams, nevents_delta):
        self.CallbackDelegate.updateReceived(worker.WID, hists, streams, nevents_delta)
        
    @synchronized
    def dataReceived(self, worker, events_delta, data):
        self.CallbackDelegate.dataReceived(worker.WID, events_delta, data)
        
    @synchronized
    def exceptionReceived(self, worker, info):
        self.CallbackDelegate.exceptionReceived(worker.WID, info)
        
    @synchronized
    def messageReceived(self, worker, nevents, message):
        self.CallbackDelegate.messageReceived(worker.WID, nevents, message)
        
    @synchronized
    def dataLoadFailureReceived(self, worker, rgid):
        self.CallbackDelegate.dataLoadFailureReceived(worker.WID, rgid)
        
    @synchronized
    def workerExited(self, worker, status, t):
        del self.WorkerInterfaces[worker.WID]
        self.DoneWorkers[worker.WID] = worker
        self.CallbackDelegate.workerExited(worker.WID, status, t, worker.NEvents, self.nrunning())
        self.log("workerExited(%s, %s). %d still running: %s" % (worker.WID, status, self.nrunning(),
            ','.join(["%s:%s" % wi.WorkerAddress for i, wi in sorted(self.WorkerInterfaces.items())])
        ))
        if self.nrunning() <= 0:    
            self.wakeup()

    def nrunning(self):
        return len(self.WorkerInterfaces)

    @synchronized
    def wait(self, timeout=None):
        t0 = time.time()
        while self.nrunning() > 0 and (timeout is None or time.time() - t0 < timeout):
            #print "Contract.wait: nrunning = ", self.nrunning()
            with self.T["Contract.wait()/loop"]:
                dt = None if timeout is None else max(0.0, t0+timeout - time.time())
                if dt is None:  dt = 1.0
                #print "Contract.wait: await(%s)" % (dt, )
                self.await(dt)
            
    @synchronized
    def abort(self):
        for w in self.WorkerInterfaces.values():
            w.abort()
            
