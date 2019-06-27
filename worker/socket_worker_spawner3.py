from socket import socket, AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR, gethostbyaddr
import json, time, multiprocessing, os, traceback, sys

from striped.client import StripedClient
from striped.common import WorkerRegistryPinger, MyThread, DXMessage, WorkerRequest, DataExchangeSocket

from SocketWorker2 import SocketWorkerBuffer 
#from sandbox import sandbox_import_module
from StripedWorker2 import WorkerDriver
from striped.common.exceptions import StripedNotFoundException

class SocketWorkerServer(multiprocessing.Process):

    def __init__(self, port, striped_server_url, module_storage, registry_address, tag, logfile, cache_limit):
        multiprocessing.Process.__init__(self)
        self.Client = StripedClient(striped_server_url, cache="long", cache_limit=cache_limit, log=self.log)
        self.Sock = socket(AF_INET, SOCK_STREAM)
        if False:
            self.Sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            self.Sock.bind(("", port))
        else:
            self.Sock.bind(("", 0))
            port = self.Sock.getsockname()[1]
        self.Port = port
        self.Sock.listen(10)
        self.ModuleStorage = module_storage
        self.Pinger = WorkerRegistryPinger(registry_address, port, tag)
        self.Stop = False
        self.WorkerModules = {}         # {module_name -> time_accessed}
        self.LogFile = None
        if logfile != None:
            logfile = "%s.%d.log" % (logfile, port)
            self.LogFile = open(logfile, "w")
        
    def log(self, msg):
        msg = "Worker %d port=%d %s: %s" % (os.getpid(), self.Port, time.ctime(time.time()), msg)
        print msg
        if self.LogFile is not None:
            self.LogFile.write(msg+"\n")
            self.LogFile.flush()
                    
    def run(self):
        signal.signal(signal.SIGINT, self.sigint)
        self.Pinger.start()
        while not self.Stop:
            self.log("accepting...")
            sock, addr = self.Sock.accept()
            dxsock = DataExchangeSocket(sock)
            #print "Client connected: %s" % (addr,)
            self.log("Client connected: %s" % (addr,))
            
            # read job description JSON
            #print "reading params..."
            
            try:    msg = dxsock.recv()
            except:
                self.log("Can not read initial message. Closing the connection. Error:\n%s" % (traceback.format_exc(),))
                msg = None
            jid = None  
            wid = None 
            if msg and msg.Type == 'request':
                try:
                    params = WorkerRequest.fromDXMsg(msg)
                    signature, t, salt, alg = msg["worker_authenticator"].split(":")
                    #print "worker_authenticator:", (signature, t, salt, alg)
                    key = self.Pinger.Key
                    verified, reason = params.verifySignature(key, signature, t, salt, alg)
                    #self.log("worker request verification: %s %s" % (verified, reason))
                    if not verified:
                        self.log("Signature verification failed: %s" % (reason,))
                        dxsock.send(DXMessage("exception").append(info="Authentication failed: %s" % (reason,)))
                    else:
                        jid, wid = params.JID, params.WID

                        try:    
                            self.runWorker(params, dxsock, addr)
                        except StripedNotFoundException as e:
                            dxsock.send(DXMessage("exception").append(info=str(e)))
                        except:
                            formatted = traceback.format_exc()
                            self.log("jid/wid=%s/%s: SocketWorkerServer.runWorker() exception:\n%s" % (jid, wid, formatted,))
                            dxsock.send(DXMessage("exception").append(info=formatted))
                except:
                    self.log("jid/wid=%s/%s: Error processing the request. Closing the connection\n%s" % (jid, wid, traceback.format_exc(),))
            dxsock.close()
            self.log("jid/wid=%s/%s: socket closed" % (jid, wid))
            
    def sigint(self):
        self.Stop = True
        self.Pinger.stop()
            
    def runWorker(self, params, dxsock, client_address):
        #buffer = AsynchronousSocketWorkerBuffer(sock)       #SocketWorkerBuffer(sock)
        #print "runWorker: HDescriptors:", params.HDescriptors
        buffer = SocketWorkerBuffer(dxsock, params.HDescriptors, log=self.log)
        module_name = "%s_%s" % (params.WorkerModuleName, os.getpid())
        worker_module = sys.modules.get(module_name)
        module_path = None
        
        #print "worker text=", params.WorkerText
        
        if worker_module is None:
            module_path = "%s/%s.py" % (self.ModuleStorage, module_name)
            open(module_path, "w").write(params.WorkerText)
            #worker_module = sandbox_import_module(module_name, ["Worker"])
            worker_module = __import__(module_name, {}, {}, ["Worker"])
        
        self.WorkerModules[module_name] = time.time()
        
        #reload(worker_module)
        worker_class = worker_module.Worker
        dataset_name = params.DatasetName
        rgids = params.RGIDs
        wid = params.WID
        user_params = params.UserParams
        use_data_cache = params.UseDataCache
        jid = params.JID

        self.log("request from %s (%s): jid/wid=%s/%s, dataset=%s, %d frames" % (client_address, gethostbyaddr(client_address[0])[0], 
                                jid, wid, dataset_name, len(rgids)))      
                                
        data_mod_client = None
        if params.DataModURL is not None and params.DataModToken is not None:
            data_mod_client = StripedClient(params.DataModURL, data_modification_token=params.DataModToken)
        worker = WorkerDriver(jid, wid, self.Client, worker_class, dataset_name, rgids, params.NWorkers, buffer, 
                user_params, use_data_cache, 
                data_mod_client,
                self.log)
        nevents = worker.run()
        self.log("jid/wid=%s/%s: worker.run() ended with nevents=%s" % (jid,wid,nevents))
        
        buffer.close(nevents)
        if module_path: 
            #print "removing %s" % (module_path,)
            os.unlink(module_path)
            #print "deleted %s" % (module_path,)
            if module_path.endswith(".py"):
                #print "removing %s" % (module_path+"c",)
                os.unlink(module_path+"c")
        self.purgeWorkerModules()
        self.log("jid/wid=%s/%s: exit from runWorker" % (jid,wid))


    ModulePurgeInterval = 30.0*60       # 30 minutes
        
    def purgeWorkerModules(self):
        for m, t in self.WorkerModules.items()[:]:
            if t < time.time() - self.ModulePurgeInterval:
                del self.WorkerModules[m]
                del sys.modules[m]
        
    def stop(self):
        self.terminate()
        
    
class WorkerSpawner:

    def __init__(self, server_url, nworkers, port_range_start, module_storage, registry_address, tag, logfile, cache_limit):
        self.Workers = []
        self.StripedServerURL = server_url
        self.NWorkers = nworkers
        self.PortRange = range(port_range_start, port_range_start+nworkers)
        self.ModuleStorage = module_storage
        self.RegistryAddress = registry_address
        if not module_storage in sys.path:
            sys.path.insert(0, module_storage)
        self.LogFile = logfile
        self.Tag = tag
        self.CacheLimit = cache_limit
        
    def run(self):
        signal.signal(signal.SIGINT, self.sigint)
        for i, p in enumerate(self.PortRange):
            w = SocketWorkerServer(p, self.StripedServerURL, self.ModuleStorage, self.RegistryAddress, self.Tag, self.LogFile,
                    self.CacheLimit)
            w.daemon = True
            self.Workers.append(w)
            w.start()
        nrunning = self.NWorkers
        dead = []
        while nrunning > 0:
            time.sleep(1)
            nrunning = 0
            for w in self.Workers:
                if w.is_alive():
                    nrunning += 1
                else:
                    if not w in dead:
                        print "ERROR: Worker %d died with status %d" % (w.pid, w.exitcode)
                        dead.append(w)
                    
    def sigint(self, signum, frame):
        print "SIGINT received. Terminating..."
        self.terminate()
            
    def terminate(self):
        for w in self.Workers:
            w.stop()

if __name__ == '__main__':
    import sys, os, signal, getopt
    
    Usage = """python socket_worker_spawner.py [-l <logfile>] [-c <cache limit, GB>] <striped server URL> <module storage> <starting port> <nworkers> <registry host> <registry port>"""
    opts, args = getopt.getopt(sys.argv[1:], "h?l:t:c:")
    opts = dict(opts)
    logfile = opts.get("-l")
    tag = opts.get("-t", "default")
    cache_limit = float(opts.get("-c", 1.0))*1.0e9
                
    if len(args) != 6 or "-h" in opts or "-?" in opts:
        print
        print Usage
        print
        sys.exit(1)
    

    server_url, module_storage, port, nworkers, registry_host, registry_port = args
    port, registry_port, nworkers = int(port), int(registry_port), int(nworkers)
    
    spawner = WorkerSpawner(server_url, nworkers, port, module_storage, (registry_host, registry_port), tag, logfile, cache_limit)
    open("socket_worker_spawner.pid", "w").write("%d" % (os.getpid(),))
    
    spawner.run()

