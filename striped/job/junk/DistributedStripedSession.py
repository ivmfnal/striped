from ..client import StripedClient
import sys, os, time, getopt, json, multiprocessing, signal
import numpy as np
from ..common import Lockable, MyThread, synchronized, Queue
from threading import Condition
from cStringIO import StringIO
import cPickle
from .StripedJob import StripedJob
import socket, traceback, random, time
from ..common import WorkerRegistryClient
from .SocketGangContract2 import SocketGangContract
from ..common import JobTracer as JT

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    import pylab as pl
except:
    pass

class DistributedStripedSession(object):

    def __init__(self, data_server_url, worker_registry_url, nworkers=None):
        self.SessionID = self.generateSessionID()
        self.RegistryURL = worker_registry_url
        self.DataServerURL = data_server_url
        self.NWorkers = nworkers
        self.IPython = None
        try:    self.IPython = __IPYTHON__
        except NameError:   self.IPython = False
        
    def generateSessionID(self):
        t = time.time()
        return "%f" % (t,)
        
    def workers(self):
        return WorkerRegistryClient(self.RegistryURL).workers(n=self.NWorkers, sid=self.SessionID)
        
    def findWorkerClass(self, headline):
        import __main__
        import re
        
        latest_def = ""
        latest_n = -1
        latest_k = None

        pattern = re.compile("^_i\d+")

        for k in dir(__main__):
            if pattern.match(k):
                v = getattr(__main__, k)
                #print "----- %s -----" % (k,)
                #print v
                if isinstance(v, (str, unicode)) and v.startswith(headline):
                    n = int(k[2:])
                    if n > latest_n:
                        v = v.split("\n", 1)[-1]
                        latest_k = k
                        latest_n = n
                        latest_def = v
        
        return latest_def
        
    def createJob(self, dataset_name, worker_class_tag = "#__worker_class__", fraction = None,
            histograms = [],
            worker_class_source = None, worker_class_file = None, display=False,
                user_callback = None, user_params = {}):
                
        assert user_params is None or isinstance(user_params, dict), "User parameters must be either None or a dictionary. %s used" % (type(user_params),)

        if worker_class_source is None and worker_class_file is not None:
                if isinstance(worker_class_file, (str, unicode)):
                        worker_class_file = open(worker_class_file, "r")
                worker_class_source = worker_class_file.read()

        if not worker_class_source and self.IPython:
                worker_class_source = self.findWorkerClass(worker_class_tag)
        
        #print "user_job_object:", user_job_object
        workers = self.workers()
        if not workers:
            raise ValueError("No workers found")
            
        if not display:
            job = DistributedBatchJob(self.DataServerURL, dataset_name, workers, user_callback, worker_class_source, user_params, fraction = fraction)
        
        elif self.IPython:
            job = DistributedIPythonJob(self.DataServerURL, dataset_name, workers, user_callback, worker_class_source, user_params, fraction = fraction)
        else:
            job = DistributedPythonJob(self.DataServerURL, dataset_name, workers, user_callback, worker_class_source, user_params, fraction = fraction)
        for h in histograms:
            job.addHistogram(h)
        return job
                    
class DistributedJob(StripedJob):

    def __init__(self, data_server_url, dataset_name, workers, user_callback_delegate, worker_class_source, user_params, fraction = None):
        StripedJob.__init__(self, data_server_url, dataset_name, user_callback_delegate, user_params)

        self.IPythonDisplay = None
        self.WorkerText = worker_class_source   
        
        self.WorkerAddresses = workers[:]
        self.DataReceivedCounter = 0
        self.LastDataReceivedCounter = 0
        self.Figure = None
        self.Animated = None
        self.EventsProcessed = 0
        self.TStart = None
        self.Contract = None
        self.EventsFraction = fraction
        self.Interrupted = False

    def start(self):
        JT.reset()
        nworkers = len(self.WorkerAddresses)

        self.Contract = SocketGangContract(self.URLHead, self.DatasetName, self.EventsFraction, nworkers,
                self.WorkerAddresses, self.WorkerText, self, self.histograms(), self.UserParams)

        self.setupDisplay()


        self.Contract.start()
        self.TStart = time.time()
        
        #self.setupDisplay(animate)
        
    @synchronized
    def refresh(self, iframe):
        #print "DistributedJob:refresh"
        # will be called to update display either by pyplot animation or by self if pyplot animation is not used
        #print "refresh"
        #print "refresh: %d:%d" % (self.LastDataReceivedCounter, self.DataReceivedCounter)
        if self.LastDataReceivedCounter >= self.DataReceivedCounter:    return
        #print "refreshing..."
        self.LastDataReceivedCounter = self.DataReceivedCounter
        self.refreshDisplay(iframe)

    def sigint(self, signum, frame):
        self.Interrupted = True
        if self.Contract is not None:
            self.Contract.abort()

    def waitDone(self):
        old_handler = signal.signal(signal.SIGINT, self.sigint)
        try:
            while not self.Finished:
                self.Contract.wait(1.0)
                if not self.Animated:       # animation mechanism will call the refresh()
                    self.refresh(1)
        finally:
            signal.signal(signal.SIGINT, old_handler)            

    def run(self):
        self.start()
        return self.waitDone()
            
        
class DistributedIPythonJob(DistributedJob):


    def setupDisplay(self):
        from IPython import display
        self.Figure = pl.figure(figsize=(10,10))
        self.DataCallbackDelegate.initDisplay(self.Figure, True)
        self.Animated = False
        self.IPythonDisplay = display
            

    def refreshDisplay(self, iframe):
        #print "DistributedIPythonJob.refreshDisplay"
        self.updateDisplay(iframe)
        self.IPythonDisplay.clear_output(wait=True)
        self.IPythonDisplay.display(pl.gcf())
        #print "display refreshed"
        pass
            
        
class DistributedPythonJob(DistributedJob):

    def setupDisplay(self):
        self.Figure = plt.figure()
        animate = self.DataCallbackDelegate.initDisplay(self.Figure, False)
        self.Animated = animate
        if self.Animated:
            a = anim.FuncAnimation(self.Figure, self.refresh, interval=1000)
        plt.show()

    def refreshDisplay(self, iframe):
        #print "refreshDisplay..."
        self.updateDisplay(iframe)

class DistributedBatchJob(DistributedJob):

    def setupDisplay(self):
        pass       

    def refreshDisplay(self, iframe):
        pass
