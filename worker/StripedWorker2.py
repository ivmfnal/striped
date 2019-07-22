from QData2 import Dataset
from striped.pythreader import PyThread, DEQueue, synchronized

import os, traceback
import numpy as np
from striped.common import Tracer
from striped.common.exceptions import StripedNotFoundException

#from sandbox import sandbox_call

class AsynchronousFrameFetcher(PyThread):

    PREFETCH_MAX = 5

    def __init__(self, driver, client, dataset_name, rgids, columns, trace=None, log=None, use_data_cache = True):
        PyThread.__init__(self)
        
        #print "AsynchronousFrameFetcher(%s, %s, %s, %s, %s)" % (driver, client, dataset_name, rgids, columns)
        
        self.Driver = driver
        #print "AsynchronousFrameFetcher: dataset: %s, columns: %s" % (dataset_name, columns)
        self.ClientDataset = client.dataset(dataset_name, columns)
        if not self.ClientDataset.exists:
            raise StripedNotFoundException("Dataset %s does not exist in the database" % (dataset_name,))
        self.Columns = self.ClientDataset.columnsAndSizes(columns)
        #print "AsynchronousFrameFetcher: columns:", self.Columns
        self.Frames = DEQueue(10)
        self.RGInfos = self.ClientDataset.rginfos(rgids)
        self.Done = False
        self.Stop = False
        self.T = trace
        self.Log = log
        self.UseDataCache = use_data_cache
        
    def log(self, msg):
        if self.Log: 
            self.Log("AsynchronousFrameFetcher: %s" % (msg,))
        
    def run(self):
        try:
            for rginfo in self.RGInfos:
                if self.Stop:   break
                #self.log("stripes...")
                try:    
                    with self.T["download"]:
                        data = self.ClientDataset.stripes(self.Columns, rginfo.RGID, use_cache=self.UseDataCache)
                    #self.log("got data")
                except:
                    info = traceback.format_exc()
                    self.log("Data loader failure for rgid %d: \n%s" % (rginfo.RGID, info))      # this needs to be sent back to the job !
                    self.Frames.append(("error", rginfo, info))
                else:
                    self.Frames.append(("ok", rginfo, data))
            self.Frames.close()
        finally:
            self.Done = True    
            self.Driver = None        
    
    def queueEmpty(self):
        return len(self.Frames) == 0
        
    def stop(self):
        self.Stop = True
        
    def __iter__(self):
        return self.Frames
        
    def frames(self):
        while not self.Done or not self.queueEmpty():
            tup = self.Frames.pop()  
            if tup is None:
                break   # queue closed
            else:
                yield tup           # (status, rginfo, data)
            
class WorkerDriver:

    class JobInterface:
        def __init__(self, driver):
            self.Driver = driver

        @property
        def job_id(self):
            return self.Driver.JID
            
        def fill(self, *params, **kv):
            args = {}
            args.update(kv)
            np = len(params)
            assert np%2 == 0, "Positional arguments must come in name, value pairs"
            for i in xrange(0, np, 2):
                args[params[i]] = params[i+1]
            self.Driver.fill(args)
            
        def send(self, *params, **kv):
            args = {}
            args.update(kv)
            np = len(params)
            assert np%2 == 0, "Positional arguments must come in name, value pairs"
            for i in xrange(0, np, 2):
                args[params[i]] = params[i+1]
            self.Driver.send(args)
            
        def message(self, text):
            self.Driver.message(text)
            
    class DatabaseInterface(object):
    
        def __init__(self, client, dataset_name, data_mod_client):
            self.Client = client
            self.DatasetName = dataset_name
            self.DataModClient = data_mod_client
            self.FrameID = None
            
        def data(self, key, dataset_association = False, as_json = False):
            dataset = self.DatasetName if dataset_association else None
            try:    data = self.Client.standaloneData(key, dataset, as_json=as_json)
            except KeyError as e:
                #print e
                raise KeyError("Data not found for key '%s', dataset_association=%s" % (key, dataset_association))
            return data
            
        def updateStripes(self, data_dict):
            if self.DataModClient is None:
                raise ValueError("Data modification is not authorized")
            self.DataModClient.dataset(self.DatasetName).putStripes(self.FrameID, data_dict)
                
        def __getitem__(self, key):
            return self.data(key)
            
            
    def __init__(self, jid, wid, client, worker_class, dataset_name, rgids, nworkers, buffer, user_params, bulk_data, use_data_cache, 
                data_mod_client, tracer = None, log = None):
        self.JID = jid
        self.Client = client
        self.DataModClient = data_mod_client
        self.WorkerClass = worker_class
        self.DatasetName = dataset_name
        self.MyID = wid
        self.NWorkers = nworkers
        self.Buffer = buffer
        self.RGIDs = rgids
        self.UserParams = user_params
        self.BulkData = bulk_data
        self.T = tracer or Tracer()
        self.Buffer.set_trace(self.T)
        self.UseDataCache = use_data_cache
        self.Log = log
        self.SeenEvents = 0
        
    def log(self, msg):
        if self.Log: 
            self.Log("WorkerDriver: %s" % (msg,))
        
    def run(self):
        T = self.T
        self.SeenEvents = 0
        job_interface = self.JobInterface(self)
        dbinterface = self.DatabaseInterface(self.Client, self.DatasetName, self.DataModClient)
        with T["worker_constructor"]:
                worker = self.WorkerClass(self.UserParams, self.BulkData, job_interface, dbinterface)
                #print "WorkerDriver.run(): worker created: %s %s, columns: %s" % (type(worker), worker, worker.Columns)
                columns = worker.Columns
        worker.Trace = T
        fetcher = AsynchronousFrameFetcher(self, self.Client, self.DatasetName, self.RGIDs, columns, trace=T, log = self.Log,
                use_data_cache = self.UseDataCache)
        fetcher.start()

        dataset = Dataset(self.Client, self.Buffer, self.DatasetName,  columns, trace=T)
            
        self.Dataset = dataset
        
        events_delta = 0
        nframes = len(self.RGIDs)
        first_frame = True
        for iframe, (status, rginfo, data) in enumerate(fetcher.frames()):
            if status == "ok":
                with T["frame"]:
                    frame = dataset.frame(rginfo, data)
                    event_group = frame.eventGroup(iframe, nframes)
                    self.SeenEvents += frame.NEvents        # update this before worker.run() so that 
                                                            # emitted data can be correctly "timestamped" with the nevents
                    out = None
                    stop = False
                    if hasattr(worker, "frame"):
                        with T["frame/worker"]:
                            #sandbox_call(worker.run, event_group, job_interface, dbinterface)
                            dbinterface.FrameID = event_group.rgid
                            stop = False
                            try:
                                out = worker.frame(event_group)
                                stop = (out == "stop")
                            except StopIteration:
                                stop = True
                    self.Dataset.ProcessedEvents = self.SeenEvents
                    self.Buffer.endOfFrame(self.SeenEvents)
                    events_delta += frame.NEvents 
                    if out:
                        with T["frame/sendData"]:
                            self.Buffer.sendData(events_delta, out)
                            events_delta = 0
                    if stop:
                        break
            else:
                self.Buffer.dataLoadFailure(rginfo.RGID, data)


        if hasattr(worker, "end"):
                out = worker.end()
                if out:
                    with T["frame/sendData"]:
                        self.Buffer.sendData(events_delta, out)
                
        fetcher.stop()
        self.log("Worker %s: done, %d events processed" % (os.getpid(), self.Dataset.ProcessedEvents))
        return self.Dataset.ProcessedEvents
        
    def fill(self, dict_items):
        with self.T["frame/worker/fill"]:
            self.Buffer.fillHistos(self.SeenEvents, dict_items)

    def send(self, dict_items):
        with self.T["frame/worker/send"]:
            self.Buffer.addStreams(self.SeenEvents, dict_items)
            
    def message(self,message):
        with self.T["frame/worker/message"]:
            self.Buffer.message(self.SeenEvents, message)
        

