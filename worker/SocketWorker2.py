import numpy as np
import socket, json, sys, traceback, time, os, random
from striped.hist import HCollector
from striped.common import  DXMessage
from striped.pythreader import PyThread, synchronized
from striped.common import encodeData
from BulkStorage import BulkStorage

class StreamBuffer(object):

    def __init__(self):
        self.Data = []

    def addData(self, nevents, data):
        self.Data.append(data)          # nevents ignored for now
        
    def flush(self):
        out = self.Data
        self.Data = []
        return out

class SocketWorkerBuffer(object):

    def __init__(self, bid, dxsock, hdescriptors, maxn=40000, trace = None, log=None):
        self.ID = bid
        self.N = 0
        self.MaxN = maxn
        self.DXSock = dxsock
        self.EventsProcessed = 0
        #self.LastEventsProcessed = 0
        self.SBuffers = {}
        self.HCollectors = { hid: HCollector(desc) 
                                for hid, desc in hdescriptors.items() }
        self.HInputs = set()
        for h in self.HCollectors.values():
            self.HInputs |= h.inputs()
        
        self.FlushInterval = 180.0     # send at most once every 180 seconds
        self.LastFlush = time.time()
        self.T = self.set_trace(trace)
        self.Log = log
        self.DataSequence = 0
        self.NFills = 0
        
    def log(self, msg):
        if self.Log is not None:
            self.Log("SocketWorkerBuffer: %s" % (msg,))

    def set_trace(self, t):
        self.T = t
        for hc in self.HCollectors.values():
            hc.T = t

    def dataLoadFailure(self, rgid, info):
        self.DXSock.send(DXMessage("data_load_failure", rgid=rgid, info=info))
        
    def setEventsProcessed(self, n):
        self.EventsProcessed = n
    
    def fillHistos(self, nevents, dct):
        input_set = set(dct.keys())
        for hc in self.HCollectors.values():
            hc_inputs = hc.inputs()
            if input_set >= hc_inputs:
                hc.fill(dct)
                self.NFills += 1
                
    def message(self, nevents, message):
        self.DXSock.send(DXMessage("message", nevents=nevents).append(message=message))
        
    def endOfFrame(self, nevents):
        #self.log("end of frame")
        t = time.time()
        if self.NFills and random.random() < (t-self.LastFlush)/self.FlushInterval:
            self.LastFlush = t
            self.flushAll(nevents)
            
    def flushAll(self, nevents):
        #self.log("flush all, nevents=%d" % (nevents,))
        
        for hid, hb in self.HCollectors.items():
            dump = hb.dump()
            #print "counts:", counts
            self.DXSock.send(DXMessage("hist", type="h", hid=hid).append(dump=dump))
            
        for sn, sb in self.SBuffers.items():
            values = sb.flush()
            if values is not None and len(values) > 0:
                # values is a list of (nevents, data)
                #print "flushAll: len(values)=%d, len(values.data)=%d" % (len(values), len(values.data))
                msg = DXMessage("stream", name=sn, format="encode")(data=encodeData(values))
                self.DXSock.send(msg)
                
        if self.NFills > 0:
            self.DXSock.send(DXMessage("flush", nevents=nevents))
        self.NFills = 0
        #self.log("flush all done")
        
    def sendData___(self, events_delta, data):
        msg = DXMessage("data", events_delta=events_delta, format="encode")(data=encodeData(data))
        self.DXSock.send(msg)

    def dataSequence(self):
            seq = self.DataSequence
            self.DataSequence += 1
            return seq

    def sendData(self, events_delta, data):
        storage_name = "%s_%s" % (self.ID, self.dataSequence())
        storage = BulkStorage.create(storage_name, data)
        msg = DXMessage("data", events_delta=events_delta, storage=storage_name)
        self.DXSock.send(msg)

    def bumpEvents(self, events_delta):
        if events_delta > 0:
             self.DXSock.send(DXMessage("events", delta=events_delta))
        
    def close(self, nevents):
        self.log("close")
        self.flushAll(nevents)
        #print "Buffer: closing connection, nevents:", nevents
     
