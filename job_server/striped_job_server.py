from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR, gethostname, gethostbyname, gethostbyaddr
from striped.pythreader import PyThread, TaskQueue, Task, Primitive, synchronized
from striped.common import DXMessage, JobDescription, DataExchangeSocket, Tracer, LogFile, random_salt
from striped.common.signed_token import SignedToken, generate_secret
from striped.common import rfc2617, encodeData, decodeData, import_from_text
from striped.client import StripedClient
from registry import RegistryServer
import multiprocessing, uuid, time, os, sys, traceback, json, yaml
from Contract import Contract
from webpie import WebPieApp, WebPieHandler, app_synchronized, Response, HTTPServer


try:
    from setproctitle import setproctitle
except:
    setproctitle = lambda x: x
    

class   Configuration(Primitive):

    def __init__(self, path=None, envVar=None):
        Primitive.__init__(self)
        # config is supposed to be ConfigParser object after the .read() is done
        self.Path = path or os.environ[envVar]
        self.load()

    @synchronized        
    def load(self):
        self.Config = yaml.load(open(self.Path, "r").read())
        self.Users = self.Config["Users"]      # { username -> { password: ..., dns: [...], roles: [...] } } 
        self.Passwords = { user:info.get("password") for user, info in self.Users.items() }
        self.Roles = { user:info.get("roles",[]) for user, info in self.Users.items() }
        self.DNs = { user:info.get("dns",[]) for user, info in self.Users.items() }
        
        self.MaxJobs = self.Config.get("MaxJobs", 5)
        self.QueueCapacity = self.Config.get("QueueCapacity", 15)
        self.Port = self.Config.get("Port", 8765)
        self.Host = self.Config.get("Host", gethostbyname(gethostname()))
        self.RegistryPort = self.Config.get("RegistryPort", 7555)
        self.DataServerURL = self.Config["DataServerURL"]                   # required
        self.SourceArchive = self.Config.get("SourceArchive")
        self.LogFileDir = self.Config.get("LogFileDir")
        self.WebServerPort = self.Config["WebServerPort"]
        self.BulkTransportPort = self.Config.get("BulkTransportPort", 1234)

    @synchronized
    def dns(self, username):
        return self.DNs.get(username, [])
        
    @synchronized
    def roles(self, username):
        return self.Roles.get(username, [])
    
    @synchronized
    def password(self, username):
        return self.Passwords.get(username)
    
    def __getitem__(self, key):     # for required parameters
        return self.Config[key]
        
    def get(self, key, default = None):
        return self.Config.get(key, default)
            
def log(text):
    if log_file is None:
        print(text)
    else:
        log_file.log(text)

class JobProcess(multiprocessing.Process):
    def __init__(self, jid, data_server_url, bulk_transport_port, dx, data_client, workers, job_description, log_file_path):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self.JID = jid
        self.DataExchange = dx
        self.Workers = workers              # list of WorkerInfo objects
        self.JobDesc = job_description
        self.DataClient = data_client
        self.TotalEvents = 0
        self.T = Tracer()
        self.ContractStartedT = None
        self.FirstWorkerExitT = self.LastWorkerExitT = None
        self.DataServerURL = data_server_url
        self.Contract = None
        self.LogFile = None
        self.LogFilePath = log_file_path
        self.BulkTransportPort = bulk_transport_port
                
    def log(self, msg):
        print(("%s: %s" % (time.ctime(time.time()), msg)))
        if self.LogFile is not None:
            self.LogFile.write("%s: %s\n" % (time.ctime(time.time()), msg))
            self.LogFile.flush()
        
    def run(self):

        self.log("job process %s started" % (self.JID,))

        if self.LogFilePath is not None:
            self.LogFile = open(self.LogFilePath, "w")
            
        try:
            with self.T["JobProcess/run"]:
                    setproctitle("striped_job %s" % (self.JID,))
                    self.log("started: dataset: %s, fraction: %s, %d workers" % (self.JobDesc.DatasetName, self.JobDesc.Fraction, len(self.Workers)))
                    callback_delegate = self
                    with self.T["JobProcess/run/create_contract"]:
                        self.Contract = Contract(self.JID, self.DataServerURL, self.BulkTransportPort, 
                            self.DataClient.dataset(self.JobDesc.DatasetName), 
                            self.JobDesc,
                            self.Workers, callback_delegate, 
                            self.log, self.T)

                    self.DataExchange.send(DXMessage(
                            "job_started", 
                            nworkers = len(self.Workers),
                            jid = self.JID,
                            total_events = self.Contract.TotalEvents,            
                            selected_events = self.Contract.SelectedEvents,
                            selected_frames = json.dumps(self.Contract.SelectedFrames)
                        )
                    )          

                    self.log("job_started sent")

                    with self.T["JobProcess/run/start_contract"]:
                        self.Contract.start()

                    self.ContractStartedT = self.FirstWorkerExitT = self.LastWorkerExitT = time.time()

                    self.log("contract started. waiting...")

                    with self.T["JobProcess/run/wait_contract"]:
                        self.Contract.wait()
                        
                    self.DataExchange.send(DXMessage("job_done", total_events = self.TotalEvents))

                    self.log("Job finished. Worker exit timestamps: first: %.5f, last:%.5f" % (self.FirstWorkerExitT-self.ContractStartedT, 
                                self.LastWorkerExitT-self.ContractStartedT))
                    self.DataExchange.close()
                    self.log("---- exit ----")
        except:
            tb = traceback.format_exc()
            self.DataExchange.send(DXMessage("job_failed").append(reason=tb))
            self.log("Exception: ------------\n%s" % (tb,))
        finally:
            self.log("----- job stats: -----\n"+self.T.formatStats())            
            if self.LogFile is not None:    self.LogFile.close()
        
    def updateReceived(self, wid, hists, streams, nevents_delta):
    
        self.TotalEvents += nevents_delta
        client_disconnected = False

        if hists:
            msg = DXMessage("histograms", total_events = self.TotalEvents, wid=wid)
            for k, v in hists.items():
                msg[k] = v
            try:    self.DataExchange.send(msg)
            except: 
                self.log("Error sending message to the client:\n%s" % (traceback.format_exc(),))
                client_disconnected = True

        if streams:
            for k, data in streams.items():
                msg = DXMessage("stream", name=k, format="pickle", total_events = self.TotalEvents, wid=wid)
                msg.append(data=data)          # this is still pickled data because the WorkerInterface does not unpickle
                try:    self.DataExchange.send(msg)
                except: 
                    self.log("Error sending message to the client:\n%s" % (traceback.format_exc(),))
                    client_disconnected = True
                
        if not streams and not hists:
            #print "sending empty(%d)" % (self.TotalEvents,)
            msg = DXMessage("empty", total_events = self.TotalEvents, wid=wid)
            try:    self.DataExchange.send(msg)
            except: 
                self.log("Error sending message to the client:\n%s" % (traceback.format_exc(),))
                client_disconnected = True
        if client_disconnected:
            self.log("Client disconnected (because of the communication error). Aborting")
            self.Contract.abort()

    def forward(self, msg):
        with self.T["callback/forward/%s" % (msg.Type,)]:
            self.DataExchange.send(msg)

    def eventsDelta(self, wid, events_delta):
        with self.T["callback/eventsDelta"]:
            self.DataExchange.send(DXMessage("events", wid=wid, events_delta=events_delta))

    def dataReceived(self, wid, events_delta, data):
        with self.T["callback/data"]:
            self.DataExchange.send(DXMessage("data", wid=wid, events_delta=events_delta).append(data=data))

    def exceptionReceived(self, wid, info):
        with self.T["callback/exception"]:
            self.DataExchange.send(DXMessage("exception", wid=wid).append(info=info))
        
    def messageReceived(self, wid, nevents, message):
        with self.T["callback/message"]:
            self.DataExchange.send(DXMessage("message", wid=wid, nevents=nevents).append(message=message))
        

    def dataLoadFailureReceived(self, wid, rgid):
        with self.T["callback/data_load_failure"]:
            self.DataExchange.send(DXMessage("data_load_failure", wid=wid, rgid=rgid))
        
    def workerExited(self, wid, status, t, nevents, nrunning):
        if self.FirstWorkerExitT is None:
            self.FirstWorkerExitT = time.time()
        self.LastWorkerExitT = time.time()
        with self.T["callback/worker_exit"]:
            self.DataExchange.send(DXMessage("worker_exit", nrunning=nrunning, wid=wid, status=status,
                t=t, nevents=nevents))
        
class JobTask(Task):
    def __init__(self, server, jid, job_description, data_server_url, bulk_transport_port, data_client, dx, log_file_path):
        Task.__init__(self)
        self.Server = server
        self.DataExchange = dx
        self.JID = jid
        self.DataClient = data_client
        self.DataServerURL = data_server_url
        self.JobDescription = job_description
        self.Started = None
        self.Ended = None
        self.Failed = False
        self.Created = time.time()
        self.LogFilePath = log_file_path
        self.BulkTransportPort = bulk_transport_port
        self.log("created: username=%s dataset=%s fraction=%s" % (job_description.Username, job_description.DatasetName,
                    job_description.Fraction))
        
    def log(self, msg):
        log("[job task %s]: %s" % (self.JID, msg))
        
    def run(self):
        self.log("started: %s" % (self.JID,))
        job_description = self.JobDescription
        failed = False
        try:
            self.log("validating...")
            validated = self.Server.validate_job(job_description)
            if not validated:
                self.log("job request validation failed %s" % (repr(job_description.AuthToken),))
                self.DataExchange.send(DXMessage("job_failed").append(reason="Token validation failed"))
                self.Failed = True
                self.Server.jobFailed(self, "Token validation failed")
            else:   
                self.log("validated: token=%s identity=[%s]" % (job_description.AuthToken, job_description.Identity))
                workers = self.Server.workers(tags=job_description.WorkerTags)
                self.log("workers: %s" % ([wi.Addr for wi in workers],))
                if not workers:
                    self.log("no workers found for tags=%s" % (job_description.WorkerTags,))
                    self.DataExchange.send(DXMessage("job_failed").append(reason="No available workers found for tags=%s" % (job_description.WorkerTags,)))
                    self.Failed = True
                    self.Server.jobFailed(self, "No available forkers found for tags=%s" % (job_description.WorkerTags,))
                else:
                    process = JobProcess(self.JID, self.DataServerURL, self.BulkTransportPort, self.DataExchange, self.DataClient, 
                                workers, job_description, self.LogFilePath)
                    process.start()
                    self.Started = time.time()
                    self.Server.jobStarted(self)
                    process.join()
                    self.log("job process exited with status %s" % (process.exitcode,))
        except:
            exception = traceback.format_exc()
            self.log("failed: %s" % (exception,))
            self.Server.jobFailed(self, exception)
            self.Failed = True
        finally:
            self.Ended = time.time()
            self.DataExchange.close()
            if not self.Failed:
                self.log("ended")
                self.Server.jobEnded(self)
            self.Server = None

    def updateReceived(self, wid, data, nevents_delta):
        msg = DXMessage("update", events_delta = nevents_delta, wid=wid)
        for k, v in data:
            msg["data:"+k] = v
        msg.toSocket(self.Sock)
        
        
    def workerExited(self, worker, status, t):
        pass

class JobServer(PyThread):
    def __init__(self, host, port, worker_registry, authenticator, data_server_url, bulk_transport_port, 
                queue_capacity, max_jobs_running, source_archive, log_file_dir):
        PyThread.__init__(self)
        self.DataServerURL = data_server_url
        self.WorkerRegistry = worker_registry
        self.Sock = None
        self.Port = port
        self.Host = host
        self.Stop = False
        self.Authenticator = authenticator
        self.MaxJobs = max_jobs_running
        self.QueueCapacity = queue_capacity
        self.JobQueue = TaskQueue(max_jobs_running, capacity=queue_capacity)
        self.JIDPrefix = "%03d" % (os.getpid() % 1000,)
        self.NextJID = 1
        self.DataClient = StripedClient(data_server_url)
        self.SourceArchive = source_archive
        self.LogFileDir = log_file_dir
        self.JobHistory = []
        self.BulkTransportPort = bulk_transport_port
        
    @synchronized
    def purgeJobHistory(self):
        now = time.time()
        self.JobHistory = list(filter(
                lambda j, tmax = now - 24*3600: 
                        j.Ended and j.Ended > tmax,
                self.JobHistory
        ))

    @synchronized        
    def jid(self):
        t = "%s%04d" % (self.JIDPrefix, self.NextJID,)
        self.NextJID = (self.NextJID + 1) % 10000
        return t
        
    def log(self, msg):
        log("[server]: %s" % (msg,))
        
    def run(self):
        self.Sock = socket(AF_INET, SOCK_STREAM)
        self.Sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.Sock.bind(('', self.Port))
        self.Sock.listen(5)
        data_exchange_listener = DataExchangeSocket(self.Sock)
        
        while not self.Stop:
            data_exchange = None
            try:
                data_exchange = data_exchange_listener.accept()
                msg = data_exchange.recv()
                #print "msg:", msg.Type
                if msg and msg.Type == 'job_request':
                        job_description = JobDescription.fromDXMsg(msg)
                        exists = self.DataClient.dataset(job_description.DatasetName).exists
                        #print "exists:", exists
                        if not exists:
                            self.log("Dataset not found: %s" % (job_description.DatasetName,))
                            data_exchange.send(DXMessage("job_failed").append(reason="Dataset '%s' not found" % (job_description.DatasetName,)))
                        else:
                            jid = self.jid()
                            self.log("Job description received. Job id %s assigned" % (jid,))
                            job_log_file_path = None if self.LogFileDir is None else "%s/job_%s.log" % (self.LogFileDir, jid)
                            jt = JobTask(self, jid, job_description, self.DataServerURL, self.BulkTransportPort, 
                                    self.DataClient, data_exchange, job_log_file_path)
                            self.JobQueue << jt
                            data_exchange = None        # the job task owns it now !
                            if self.SourceArchive is not None:
                                open("%s/ws_%s.txt" % (self.SourceArchive, jid), "w").write(job_description.WorkerText)
                self.purgeJobHistory()
            except:
                dump = traceback.format_exc()
                self.log("Uncaught exception: %s" % (dump,))
                if data_exchange is not None:
                    data_exchange.send(DXMessage("job_failed").append(reason="Exception: %s" % (dump,)))
            finally:
                if data_exchange is not None:
                    data_exchange.close()
                    data_exchange = None
                    
            
    def workers(self, tags=None):
        return self.WorkerRegistry.workers(tags=tags)
        
    def validate_job(self, job_description):
        validated, identity = self.Authenticator.validate(job_description.AuthToken, job_description.Username)
        if validated:
            job_description.Identity = identity
        return validated

    @synchronized        
    def jobStarted(self, job_task):
        jid = job_task.JID
        self.log("Jobs running: " + ",".join([j.JID for j in self.JobQueue.activeTasks()]))
        
    @synchronized        
    def jobs(self):
        self.purgeJobHistory()
        queued, running = self.JobQueue.tasks()[:]
        ids = set([j.JID for j in queued + running])
        return queued, running, [j for j in self.JobHistory if not j.JID in ids]
            
    @synchronized        
    def jobEnded(self, job_task):
        self.JobHistory.append(job_task)
        jid = job_task.JID
        self.log("Jobs running: " + ",".join([j.JID for j in self.JobQueue.activeTasks() if j.JID != jid]))
        
    @synchronized        
    def jobFailed(self, job_task, reason):
        self.JobHistory.append(job_task)
        jid = job_task.JID
        self.log("Jobs running: " + ",".join([j.JID for j in self.JobQueue.activeTasks() if j.JID != jid]))

class Authenticator(Primitive):

    TTL = 24*3600         # 24 hours
    TTL_Leeway = 600
    
    def __init__(self, config):   
        Primitive.__init__(self)     
        self.Config = config
        self.AARequired = config.get("AARequired", "yes") == "yes"
        self.Secret = generate_secret(128)

    @synchronized
    def authenticate_dn(self, username, dn):
        return dn in self.Config.dns(username)

    @synchronized
    def password(self, username):
        return self.Config.password(username)
        
    @synchronized
    def authorize(self, username):
        token = SignedToken({"identity":username}, expiration=self.TTL).encode(self.Secret)
        #print("token: %s" % (token,))
        return token
        
    @synchronized
    def validate(self, token, username):
        if not self.AARequired:
            return True, None
        try:    token = SignedToken.decode(token, self.Secret, verify_times=True, leeway=self.TTL_Leeway)
        except Exception as e:
            #print ("Authenticator.validate: %s" % (traceback.format_exc(),))
            return False, str(e)
        return token.Payload.get("identity") == username, token.Payload.get("identity")
            
class WebServiceHandler(WebPieHandler):

    def __init__(self, request, app, path):
        WebPieHandler.__init__(self, request, app, path)
                
    def log(self, msg):
        log("[WebServer]: %s" % (msg,))
        
    def job_server_address(self, request, relpath, **args):
        return Response(
            json.dumps(
                [self.App.JobServer.Host, self.App.JobServer.Port]
            ),
            content_type = "text/json"
        )

    def token_digest(self, request, relpath, **args):
        ok, header = rfc2617.digest_server("run", request.environ, self.App.get_password)
        if not ok:
            if header:
                resp = Response("Authorization required", status=401)
                resp.headers['WWW-Authenticate'] = header
            else:
                self.log("authentication failed for user %s %s" % (username,))
                resp = Response("Authentication or authorization failed", status=403, content_type="text/plain")
            return resp
        username = header
        token = self.App.authorize(username)
        if not token:
            self.log("authorization failed for user %s %s" % (username,))
            return Response("Authentication or authorization failed", status=403, content_type="text/plain")
        
        return Response(token, content_type="text/plain")

    def token(self, request, relpath, method="digest", **args):
        if method == "digest":
            return self.token_digest(request, relpath, **args)
        else:
            return Response("Invalid authentication method", status=400)
                    
    def ___validate(self, request, relpath, token=None, **args):
        tup = self.App.validate(token)
        if not tup:
            return Response("400 Validation failed", status=400)
        else:
            return Response(json.dumps({
                    "token":  token,
                    "expiration": tup[0],
                    "username": tup[1],
                    "identity": tup[2]
                }),
                content_type = "text/josn"
            )
            
    def workers_json(self, request, relpath, tags=None, **args):
        tags = tags.split(",") if tags else None
        lst = self.App.workers(tags)
        out = json.dumps(lst)
        return Response(out, content_type = "text/json")
        
    def workers(self, request, relpath, **args):
        workers = sorted(self.App.workersInfo())        # this will be [(tag, ip, port, last_ping)] sorted by tag, ip, port
        #print "workers:", workers
        # translate ip addresses to host names
        
        out = []
        ip_translated = {}          # ip -> name
        workers_per_host = {}
        total_workers = 0
        for wi in workers:
            tag, ip, port, last_ping = wi.Tag, wi.Addr[0], wi.Addr[1], wi.LastPing
            name = ip_translated.get(ip)
            if not name:
                try:    name = gethostbyaddr(ip)[0]
                except: name = ip
                ip_translated[ip] = name
            n = workers_per_host.get((tag,name),0)
            n += 1
            total_workers += 1
            workers_per_host[(tag,name)] = n
        
        return self.render_to_response("workers.html", 
            workers = [(tag, name, n) for (tag, name), n in sorted(workers_per_host.items())],
            total_workers = total_workers
        )
        
    def jobs_json(self, request, relpath, **args):
        queued, running, history = self.App.jobs()
        data = {
            "queued": [ 
                { 
                    "jid":              j.JID, 
                    "username":         j.JobDescription.Username,
                    "dataset":          j.JobDescription.DatasetName,
                    "created":          j.Created,
                    "fraction":         j.JobDescription.Fraction
                } 
                for j in queued
            ],
            "running":  [ 
                { 
                    "jid":              j.JID, 
                    "username":         j.JobDescription.Username,
                    "dataset":          j.JobDescription.DatasetName,
                    "created":          j.Created,
                    "started":          j.Started,
                    "fraction":         j.JobDescription.Fraction
                } 
                for j in running
            ],
            "history":  [
                { 
                    "jid":              j.JID, 
                    "username":         j.JobDescription.Username,
                    "dataset":          j.JobDescription.DatasetName,
                    "created":          j.Created,
                    "started":          j.Started,
                    "ended":            j.Ended,
                    "fraction":         j.JobDescription.Fraction
                } 
                for j in history
            ]
            
        }
        return Response(json.dumps(data, indent=4, sort_keys=True), content_type = "text/json")
        
    def jobs(self, request, relpath, **args):
        queued, running, history = self.App.jobs()
        return self.render_to_response("jobs.html", queued=queued, running=running, history=history)
        
    def index(self, request, relpath, **args):
        nworkers = len(self.App.workersInfo())
        return self.render_to_response("index.html",
            nworkers = nworkers,
            config = self.App.Config)
        

def delta(t, t_from=None):
    if t is None or t_from is None:  return ""
    if t_from == 0:     
        dt = time.time() - t
    else:
        dt = t - t_from
    dt = abs(dt)
    if dt < 60:
        delta_txt = "%.2fs" % (dt,)
    elif dt < 600:
        delta_txt = "%dm%ds" % (int(dt/60), int(dt)%60)
    elif dt < 3600:
        delta_txt = "%dm" % (int(dt)/60,)
    elif dt < 3600*24:
        delta_txt = "%dh" % (int(dt)/3600,)
    else:
        delta_txt = "%dd%dh" % (int(dt)/(3600*24), ((int(dt)%(3600*24))/3600))
    return delta_txt
    
def ctime(t):
    if t is None:   return ""
    return time.ctime(t)
    
def job_status(task):
    if task.Ended is not None:
        return "failed" if task.Failed else "done"
    elif task.Started is not None:
        return "running"
    else:
        return "queued"
    
        
class WebServiceApp(WebPieApp):

    def __init__(self, config, registry, authenticator, job_server):
        WebPieApp.__init__(self, WebServiceHandler)
        self.Config = config
        self.Registry = registry
        self.Authenticator = authenticator
        self.Started = False
        self.JobServer = job_server
        templates = os.environ['JINJA_TEMPLATES_LOCATION']
        self.initJinjaEnvironment(tempdirs = templates, 
            filters = {
                "ctime": ctime,
                "delta": delta,
                "job_status":job_status
                }, 
            globals = {})
        
    @app_synchronized
    def get_password(self, realm, username):
        return self.Authenticator.password(username)
        
    @app_synchronized
    def authorize(self, username):
        if "run" in self.Config.roles(username):
            return self.Authenticator.authorize(username)
        else:
            return None
            
    @app_synchronized
    def jobs(self):
        return self.JobServer.jobs()       
    
    @app_synchronized
    def workersInfo(self):
        return self.Registry.workersInfo() 

import getopt, sys

opts, args = getopt.getopt(sys.argv[1:], "c:")
opts = dict(opts)
config_file = opts.get("-c") or os.environ["STRIPED_JOB_SERVER_CFG"]    
        
config = Configuration(config_file)

# for the future: get these params from a config file
max_jobs = config.MaxJobs
queue_capacity = config.QueueCapacity
port = config.Port
host = config.Host
registry_port = config.RegistryPort
data_server_url = config.DataServerURL             # required
log_file_dir = config.LogFileDir
log_file = None
if log_file_dir:
    log_file = LogFile(log_file_dir + "/job_server.log")
source_archive = config.SourceArchive

rs = RegistryServer(registry_port)
authenticator = Authenticator(config)
js = JobServer(host, port, rs, authenticator, data_server_url, config.BulkTransportPort, 
        queue_capacity, max_jobs, source_archive, log_file_dir)
application = WebServiceApp(config, rs, authenticator, js)
web_server = HTTPServer(config.WebServerPort, application)

rs.start()
js.start()
web_server.start()
print("Web server port:", config.WebServerPort)
print("Job server port:", config.Port)
web_server.join()


