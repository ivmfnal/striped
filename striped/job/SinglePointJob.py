import sys, os, time, json, signal, stat
import numpy as np
from ..common import synchronized, Lockable
from .StripedJob import StripedJob
import socket, traceback, random, time, threading
from ..common import WorkerRegistryClient, JobDescription
from ..common.rfc2617 import digest_client
from ..common import JobTracer as JT, DXMessage, DataExchangeSocket
from ..common.signed_token import SignedToken, TokenBox

from striped.common import decodeData

import requests, yaml

class SinglePointStripedSession(Lockable):

    def _____init__(self, job_server_address=None, worker_tags = None, username=None,
                    cert_file=None, key_file=None):
        Lockable.__init__(self)
        self.SessionID = self.generateSessionID()
        if job_server_address is None:
            addr = os.environ["STRIPED_JOB_SERVER_ADDRESS"]
            host, port = addr.split(":", 1)
            port = int(port)
            job_server_address = (host, port)
        self.CertFile = cert_file or os.environ.get("X509_USER_CERTIFICATE")
        self.KeyFile = key_file or os.environ.get("X509_USER_KEY")
        self.JobServerHost, self.JobServerPort = job_server_address
        self.JobServerAddress = job_server_address
        self.AuthenticationPort = self.JobServerPort + 1
        self.WorkerTags = worker_tags
        self.IPython = None
        try:    self.IPython = __IPYTHON__
        except NameError:   self.IPython = False
        self.Username = username or os.environ["USER"]
        self.AuthToken, self.AuthTokenExpiration = None, None
        
    def __init__(self, config=None, worker_tags = None, username=None, password=None, data_modification = False):
        Lockable.__init__(self)
        self.SessionID = self.generateSessionID()
        config = config or os.environ.get("STRIPED_CLIENT_CFG")
        try:    st = os.stat(config)
        except: raise RuntimeError("Can not stat Striped client configuration file %s" % (config,))
        protected = (stat.S_IMODE(st.st_mode) % 64) == 0
        if not protected:
            raise RuntimeError("Striped client configuration file %s must be protected. Use: \n   $ chmod 0500 %s" % (config, config))
        config = yaml.load(open(config, "r"), Loader=yaml.BaseLoader)
        web_service_address = (config["JobServer"]["host"], config["JobServer"]["port"])
        worker_tags = worker_tags if worker_tags is not None else config.get("WorkerTags")
        username = username or config["Username"] 
        password = password or config["Password"]
        if web_service_address is None:
            addr = os.environ["STRIPED_JOB_SERVER_ADDRESS"]
            host, port = addr.split(":", 1)
            port = int(port)
            web_service_address = (host, port)
        if not web_service_address:
            raise ValueError("Job server address is not specified")
        if not username:
            raise ValueError("Username is not specified")
        if not password:
            raise ValueError("Password is not specified")
            
        self.WebServiceHost, self.WebServicePort = web_service_address
        
        #print web_service_address
        
        resp = requests.get("http://%s:%s/job_server_address" % (self.WebServiceHost, self.WebServicePort))
        if resp.status_code // 100 != 2:
            raise RuntimeError("Can not get Job Server addres from the web service")
        addr = tuple(json.loads(resp.text))

        self.JobServerAddress = (self.WebServiceHost, addr[1])
        
        
        #print addr

        self.WorkerTags = worker_tags
        self.IPython = None
        try:    self.IPython = __IPYTHON__
        except NameError:   self.IPython = False
        self.Username = username
        self.Password = password
        self.AuthTokenBox = TokenBox("http://%s:%s/token?role=run" % (self.WebServiceHost, self.WebServicePort),
                            self.Username, self.Password)

        self.DataModURL = None
        self.DataModUsername = None
        self.DataModPassword = None
        self.DataModTokenBox = None

        self.DataModification = data_modification   
        if data_modification:
            mod_data_server_cfg = config["DataModificationServer"]
            self.DataModURL = mod_data_server_cfg["URL"]
            self.DataModUsername = mod_data_server_cfg["Username"]
            self.DataModPassword = mod_data_server_cfg["Password"]
            self.DataModTokenBox = TokenBox("%s/token?role=upload_stripes" % (self.DataModURL,),
                            self.DataModUsername, self.DataModPassword)
            
    def renewAuthToken_x509(self):  
        # TODO: check file key protection here
        resp = requests.get("https://%s:%s/authorize?user=%s" % (self.JobServerHost, self.AuthenticationPort, self.Username),
                        cert=(self.CertFile, self.KeyFile), verify=False)
        if resp.status_code//100 == 2:
            response = json.loads(resp.text)
            return response["token"], response["expiration"], response["identity"]
        else:
            return None, None, resp.text

    def requestStripesModificationToken(self):
        status, body = digest_client("%s/token?role=upload_stripes" % (self.DataServerURL,),
                            self.DataServerUsername, self.DataServerPassword)
        if status//100 == 2:
            encoded = body.strip()
            t = SignedToken.decode(encoded)
            return encoded, t.Expirationt, t.Payload.get("identity", "")
        else:
            return None, None, body
        
    def requestAuthToken(self):
        status, body = digest_client("http://%s:%s/token?role=run" % (self.WebServiceHost, self.WebServicePort),
                            self.Username, self.Password)
        if status//100 == 2:
            encoded = body.strip()
            t = SignedToken.decode(encoded)
            return encoded, t.Expirationt, t.Payload.get("identity", "")
        else:
            return None, None, body
        
    def generateSessionID(self):
        t = time.time()
        return "%f" % (t,)
        
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
                if isinstance(v, str) and v.startswith(headline):
                    n = int(k[2:])
                    if n > latest_n:
                        v = v.split("\n", 1)[-1]
                        latest_k = k
                        latest_n = n
                        latest_def = v
        
        return latest_def
        
    @synchronized
    def createJob(self, dataset_name, worker_class_tag = "#__worker_class__", fraction = None,
            histograms = [],
            frame_selector = None,
            worker_class_text = None, worker_class_file = None, display=False,
            user_callback = None,
            callbacks = None, user_params = {}, use_data_cache = True, bulk_data = {}):

        if user_callback is not None:
            print()
            print("Deprecation warning:")
            print("'user_callback' argument of the createJob() method is being deprecated " + \
                    "and replaced with more generic 'callbacks' argument.")
            print("New 'callbacks' argument accepts either single callback object or a list of callback objects.")
            print()
            if not callbacks:   callbacks = []
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            callbacks.append(user_callback)
            
        assert user_params is None or isinstance(user_params, dict), "User parameters must be either None or a dictionary. %s used" % (type(user_params),)

        if worker_class_text is None and worker_class_file is not None:
                if isinstance(worker_class_file, str):
                        worker_class_file = open(worker_class_file, "r")
                worker_class_text = worker_class_file.read()

        if not worker_class_text and self.IPython:
                worker_class_text = self.findWorkerClass(worker_class_tag)

        assert not not worker_class_text, "Worker code must be specified"

        auth_token = self.AuthTokenBox.token
        identity = self.AuthTokenBox.Identity
        job_desc = JobDescription(dataset_name, fraction, 
                worker_class_text, user_params, frame_selector, self.WorkerTags, use_data_cache,
                auth_token, self.Username, identity, 
                self.DataModURL, self.DataModTokenBox.token if self.DataModTokenBox is not None else None,
                bulk_data)

        #print "user_job_object:", user_job_object
        if not display:
            job = SPBatchJob(job_desc, self.JobServerAddress, callbacks, use_data_cache)
        
        elif self.IPython:
            job = SPIPythonJob(job_desc, self.JobServerAddress, callbacks, use_data_cache)
        else:
            job = SPPythonJob(job_desc, self.JobServerAddress, callbacks, use_data_cache)
            
        for h in histograms:
            job.addHistogram(h)

        return job

class SinglePointJob(StripedJob):

    def __init__(self, job_desc, job_server_address, callbacks, use_data_cache):
        StripedJob.__init__(self, "", job_desc.DatasetName, callbacks, job_desc.UserParams)

        self.IPythonDisplay = None
        self.WorkerText = job_desc.WorkerText   
        self.JobServerAddress = job_server_address
        self.DataReceivedCounter = 0
        self.LastDataReceivedCounter = 0
        self.Figure = None
        self.Animated = None
        self.EventsProcessed = 0
        self.TStart = None
        self.Contract = None
        self.EventsFraction = job_desc.Fraction
        self.Interrupted = False
        self.DXSocket = None
        self.NRunning = None
        self.NWorkers = None
        self.WorkerTags = job_desc.WorkerTags
        self.UseDataCache = use_data_cache
        self.FrameSelector = job_desc.FrameSelector
        self.JobDescription = job_desc

    def start(self):
        JT.reset()

        #print "SinglePointJob: contacting job server at:", self.JobServerAddress
        self.DXSocket = DataExchangeSocket.connect(self.JobServerAddress)

        hdescriptors = {hid:h.descriptor() if not isinstance(h, dict) else h
                            for hid, h in self.histograms().items()}

        job_desc = self.JobDescription
        job_desc.addHistograms(hdescriptors)
        msg = job_desc.toDXMsg()
        msg.toExchange(self.DXSocket)
        #print "SinglePointJob: receiving message from the job server..."
        msg = self.DXSocket.recv()
        #print "    response received"
        if msg is None:
            raise RuntimeError("Server failed to acknowledge the job")
        elif msg.Type == "job_failed":
            raise RuntimeError("Job submission failed: %s" % (msg["reason"],))
        elif msg.Type != "job_started":
            raise RuntimeError("Server responded with wrong message type: %s, expected job_started" % (msg.Type,))
        self.NRunning = self.NWorkers = msg["nworkers"]
        self.EventsInDataset = msg["total_events"]
        self.EventsToProcess = msg["selected_events"]
        self.FramesToProcess = json.loads(msg["selected_frames"])
        self.JID = msg["jid"]
        self.setupDisplay()
        self.TStart = time.time()
        
        self.jobStarted()
        
        #self.setupDisplay(animate)
        
    @synchronized
    def refresh(self, iframe):
        if self.LastDataReceivedCounter >= self.DataReceivedCounter:    return
        self.LastDataReceivedCounter = self.DataReceivedCounter
        self.refreshDisplay(iframe)

    def sigint(self, signum, frame):
        self.Interrupted = True
        if self.Contract is not None:
            self.Contract.abort()

    def waitDone(self):
        old_handler = signal.signal(signal.SIGINT, self.sigint) \
            if isinstance(threading.current_thread(), threading._MainThread) \
            else None
            

        try:
            done = False
            while not done:
                msg = self.DXSocket.recv()
                if not msg:
                    done = True
                    self.jobFailed("Job server closed the connection")
                    #print "empty message"
                    break
                    
                #print ("SinglePointJob: msg type=%s" % (msg.Type,))

                if msg.Type == "hist":
                    wid = int(msg["wid"])
                    dumps = {}
                    for k, v in msg.items():
                        if k.startswith("h:"):
                            dumps[k[2:]] = v 
                    self.histogramsReceived(wid, dumps)
                    
                elif msg.Type == "stream":
                    wid = int(msg["wid"])
                    total_events = msg["total_events"]
                    name = msg["name"]
                    format = msg["format"]
                    assert format=="pickle", "Unknown stream serialization format %s" % (format,)
                    data = decodeData(msg["data"])
                    self.updateReceived(wid, {name:data}, total_events)

                elif msg.Type == "events":
                    wid = int(msg["wid"])
                    events_delta = msg.get("events_delta")
                    self.eventsDelta(wid, events_delta)
                    
                elif msg.Type == "data":
                    wid = int(msg["wid"])
                    data = decodeData(msg["data"])
                    events_delta = msg.get("events_delta")
                    self.dataReceived(wid, events_delta, data)
                    
                    
                elif msg.Type == "empty":
                    wid = int(msg["wid"])
                    total_events = msg["total_events"]
                    #print "empty %d" % (total_events,)
                    self.updateReceived(wid, {}, total_events)
                    
                elif msg.Type == "exception":
                    wid = msg["wid"]
                    info = msg["info"]
                    self.exceptionReceived(wid, info)
                    
                elif msg.Type == "message":
                    wid = msg["wid"]
                    nevents = msg["nevents"]
                    message = msg["message"]
                    self.messageReceived(wid, nevents, message)
                    
                elif msg.Type == "data_load_failure":
                    wid = msg["wid"]
                    rgid = msg["rgid"]
                    self.dataLoadFailureReceived(wid, rgid)
                    
                elif msg.Type == "job_done":
                    #self.EventsProcessed = msg["total_events"]
                    self.NRunning = 0
                    self.jobFinished()
                    done = True
                                        
                elif msg.Type == "worker_exit":
                    self.NRunning = msg["nrunning"]
                    self.workerExited(msg["wid"], msg["status"], msg["t"], msg["nevents"], msg["nrunning"])
                    #if self.NRunning <= 0:
                    #    self.jobFinished()
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
            self.destroy()      # do necessary clean-up
    
    def run(self):
        self.start()
        return self.waitDone()
        
class SPIPythonJob(SinglePointJob):


    def setupDisplay(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        import pylab as pl
        from IPython import display
        self.Figure = pl.figure(figsize=(10,10))
        self.DataCallbackDelegate.initDisplay(self.Figure, True)
        self.Animated = False
        self.IPythonDisplay = display
            

    def refreshDisplay(self, iframe):
        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        import pylab as pl
        #print "DistributedIPythonJob.refreshDisplay"
        self.updateDisplay(iframe)
        self.IPythonDisplay.clear_output(wait=True)
        self.IPythonDisplay.display(pl.gcf())
        #print "display refreshed"
        pass
            
        
class SPPythonJob(SinglePointJob):

    def setupDisplay(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as anim
        import pylab as pl
        self.Figure = plt.figure()
        animate = self.DataCallbackDelegate.initDisplay(self.Figure, False)
        self.Animated = animate
        if self.Animated:
            a = anim.FuncAnimation(self.Figure, self.refresh, interval=1000)
        plt.show()

    def refreshDisplay(self, iframe):
        #print "refreshDisplay..."
        self.updateDisplay(iframe)

class SPBatchJob(SinglePointJob):

    def setupDisplay(self):
        pass       

    def refreshDisplay(self, iframe):
        pass
