from ..client import StripedClient
import sys, os, time, getopt, json, multiprocessing, uuid, cPickle
import numpy as np
from ..common import Lockable, MyThread, synchronized, Queue
from threading import Condition
from cStringIO import StringIO
from .StripedJob import StripedJob
import socket, traceback, random, time
from ..common import WorkerRegistryClient, WorkerRequest, DXMessage
from ..common import JobTracer as JT

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import pylab as pl
except:
    pass

def distribute_items_simple(lst, n):
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
                
def dist_rec(dist, nextra):
    if nextra <= 0: return dist
    
    extra = []
    
    while len(extra) < len(dist[0]):
    
        #1. shave top floor
        n0 = len(dist[0])
        for lst in dist:
            while len(lst) > n0 and len(extra) < n0:
                extra.append(lst.pop())
            if len(extra) >= n0: break
            
        if len(extra) >= n0: break        
            
        #2. dig
        for lst in dist:
            extra.append(lst.pop())
            if len(extra) >= len(dist[0]):
                break
    return dist_rec(dist + [extra], nextra-1)

#def distribute_items(items, n):
#    return dist_rec([items], n-1)

distribute_items = distribute_items_simple

class SocketWorkerInterface(MyThread):

    def __init__(self, contract, wid, nworkerm, params, address):
        MyThread.__init__(self)
        self.WID = wid
        self.Contract = contract
        self.WorkerAddress = address
        self.Params = params
        self.TStart = None
        self.NEvents = 0
        self.LastNEvents = 0
        self.Buffer = ""
        self.Abort = False

    def abort(self):
        self.Abort = True
        
    def run(self):
        tstart = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:    sock.connect(self.WorkerAddress)
        except: 
            print ("Error connecting to %s" % (self.WorkerAddress,))
            raise

        DXMessage("request")(request=self.Params.asJSON()).toSocket(sock)
        
        eof = False
        head = ''
        data_chunks = {}
        while not eof:
	        #print "%s: readCunk..." % (self.WID,)
            
            if self.Abort:
                sock.close()
                break

            #print "DXMessage.fromSocket..."            
            msg, head = DXMessage.fromSocket(sock, head)
            if msg is None:
                eof = True
                
            else:
                
                if msg.Type == 'flush':
                    nevents = int(msg["nevents"])
                    events_delta = nevents - self.LastNEvents
                    self.LastNEvents = nevents
                    self.Contract.updateReceived(self, data_chunks, events_delta)
                    data_chunks = {}

                elif msg.Type == 'hist':
                    data_chunks[msg["hid"]] = msg["dump"]

                elif msg.Type == 'stream':
                    dtype = np.dtype(msg["dtype"])
                    name = msg["name"]
                    width = msg["width"]
                    data = msg["data"]
                    data_chunks[name] = np.frombuffer(data, dtype).reshape((-1, int(width)))
                        
        #print "Joining subprocess %d" % (self.WID,)
        #print "Worker exited"
        self.Contract.workerExited(self, 0, time.time() - tstart)
        self.Contract = None        # break circular dependencies

class SocketGangContract(Lockable):

    def __init__(self, url_head, dataset_name, events_fraction, nworkers, 
                worker_addresses, worker_text, callback_delegate, histograms, user_params):
    
        #print "SocketGangContract: worker_text=%s" % (worker_text,)
    
        Lockable.__init__(self)
        self.Workers = {}
        self.DoneWorkers = {}
        self.CallbackDelegate = callback_delegate
        self.WorkerText = worker_text
        self.DatasetName = dataset_name
        self.WorkerAddresses = worker_addresses
        self.EventsProcessed = 0
        
        hdescriptors = {hid:h.descriptor() if not isinstance(h, dict) else h
                            for hid, h in histograms.items()}
        #print hdescriptors
        
        # assign rgids

        client = StripedClient(url_head)
        dataset = client.dataset(dataset_name)        

        rgids = dataset.rgids
        if isinstance(events_fraction, float):
            nrgids = float(len(rgids)) * events_fraction
            n = int(nrgids)
            if n < nrgids:
                n += 1
            rgids = rgids[:n]
        #print "%d workers: %s" % (len(worker_addresses), worker_addresses)
        self.NWorkers = min(nworkers, len(rgids), len(worker_addresses))

        self.Params = []     # list of lists
        rgids_dist = distribute_items(rgids, self.NWorkers)

        worker_module_name = "wm_%s" % (uuid.uuid1(),)
        
        for iw in xrange(self.NWorkers):
            rgid_list = rgids_dist[iw]
            #print iw, len(rgid_list)
            params = WorkerRequest(iw, url_head, dataset_name, rgid_list, self.NWorkers, worker_module_name, worker_text, hdescriptors, user_params)
            self.Params.append(params)  
	#print "Params created"
                 
    def start(self):
        for i, params in enumerate(self.Params):
            w = SocketWorkerInterface(self, i, self.NWorkers, params, self.WorkerAddresses[i])
            self.Workers[i] = w
            w.start()
            time.sleep(0.001)
            
    @synchronized
    def nevents(self):
        n = sum(w.NEvents for w in self.Workers.values()) + sum(w.NEvents for w in self.DoneWorkers.values())
        #print "sum of workers:", n
        return n
    
    @synchronized
    def outputReceived(self, worker, key, data, nevents_delta):
        with JT["SocketGangContract.outputReceived"]:
            #print "Contract.outputReceived():", self.CallbackDelegate
            self.CallbackDelegate.outputReceived(self, worker.WID, key, data, nevents_delta)
        
    @synchronized
    def updateReceived(self, worker, data, nevents_delta):
        with JT["SocketGangContract.outputReceived"]:
            #print "Contract.outputReceived():", self.CallbackDelegate
            self.CallbackDelegate.updateReceived(worker.WID, data, nevents_delta)
        
        
    @synchronized
    def workerExited(self, worker, status, t):
        del self.Workers[worker.WID]
	#print "worker exited: %s, %d still running" % (worker.WID, len(self.Workers))
        self.DoneWorkers[worker.WID] = worker
        self.CallbackDelegate.workerExited(self, worker.WID, status, t, worker.NEvents)
        self.notify()

    @synchronized    
    def nrunning(self):
        return len(self.Workers)

    def wait(self, timeout=None):
        t0 = time.time()
        while self.nrunning() > 0 and (timeout is None or time.time() - t0 < timeout):
            dt = None if timeout is None else max(0.0, t0+timeout - time.time())
            Lockable.wait(self, dt)
            
    @synchronized
    def abort(self):
        for w in self.Workers.values():
            w.abort()
            

