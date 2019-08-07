from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_REUSEADDR, gethostbyaddr
import time, random, select
from striped.common import MyThread, synchronized, random_salt

Debug = True

def debug(msg):
    if Debug:
        print(msg)
        
class WorkerInfo:
    def __init__(self, server_addr, tag):
        self.Addr = server_addr
        self.LastPing = None
        self.Key = random_salt()
        self.Tag = tag
        
    def ping(self):
        self.LastPing = time.time()
        self.Active = True

class ___Registry(MyThread):

    def __init__(self):
        MyThread.__init__(self)
        self.Registry = {}      # {tag: {(ip,port): last ping time}}
        self.PingTimeout = 5   # seconds
        self.PurgeInterval = 12
        
    @synchronized
    def pingReceived(self, tag, ip, port):
        if Debug:
                try:    
                        hn = gethostbyaddr(ip)[0]
                except:
                        hn = ""
                debug("ping from: [%s] %s(%s):%d" % (tag, ip, hn, port))
        
        dct = self.Registry.get(tag)
        if dct is None: 
            dct = {}
            self.Registry[tag] = dct
        dct[(ip, port)] = time.time()
        #debug("added: %s/(%s,%s)" % (tag, ip, port))
        
    @synchronized
    def purge(self):
        now = time.time()
        #debug("purging %s" % (now,))
        for tag, dct in self.Registry.items():
            lst = dct.items()
            for k, t in lst:
                #debug("key/t: %s/%s" % (k, t))
                if t < now - self.PingTimeout:
                    del dct[k]
                    debug("....> purged: %s/%s" % (tag, k))

    @synchronized
    def getList(self, tags, salt, n = None):
        tag = None if tags is None else tags[0] # for now, only 1 tag, ignore the rest
        if tags is None:
            # all workers
            lst = [
                    wi
                        for dct in self.Registry.values()
                        for wi in dct.values()
                ]        
        else:
            lst = self.Registry.get(tag, []).values()
        if not lst: return []
        servers = sorted(lst, key=lambda wi: (wi.Tag, wi.Addr))           
        if n is not None and len(servers) > n:
            state = random.getstate()
            random.seed(hash(salt))
            servers = random.sample(servers, n)
            random.setstate(state)
        return servers
        
    @synchronized
    def workersInfo(self):
        #
        # returns list of workers and their tags:
        # [(tag, ip, port, last_ping)]
        #
        
        return [
            wi
                for dct in self.Registry.values()
                for wi in dct.values()
        ]
        
                
    def run(self):
        while True:
            time.sleep(self.PurgeInterval)
            self.purge()
            
class RegistryPurger(MyThread): 

    def __init__(self, server):
        MyThread.__init__(self)
        self.Server = server
        self.Interval = 1.0
        self.Stop = False
        
    def run(self):
        while not self.Stop:
            time.sleep(self.Interval)
            self.Server.purge()
        self.Server = None
            
    def stop(self):
        self.Stop = True
        
class RegistryServer(MyThread):
    # combines PingReceiver and Registry, but without HTTP server
    
    def __init__(self, port):
        MyThread.__init__(self)
        self.Port = port
        self.KnownWorkers = {}    # {(udp sock addr): WorkerInfo}
        self.Registry = {}      # {tag: {(ip,ping_port,job_port): WorkerInfo}}
        self.PingTimeout = 5    # seconds
        self.PurgeInterval = 12.0
    
    @synchronized
    def purge(self):
        now = time.time()
        #debug("purging %s" % (now,))
        for dct in self.Registry.values():
            for k, wi in dct.items():
                #debug("key/t: %s/%s" % (k, t))
                t = wi.LastPing
                if t < now - self.PingTimeout:
                    del dct[k]

        self.KnownWorkers = { k:wi for k, wi in self.KnownWorkers.items() if wi.LastPing is None or wi.LastPing > now - 3600 }

    def run(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        self.Sock = sock
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.Port))
        purger = RegistryPurger(self)
        purger.start()
        try:
            while True:
                msg, addr = sock.recvfrom(10000)
                msg = msg.decode("utf-8", "ignore")
                print("received: '%s' from %s" % (msg, addr))
                try:
                    words = msg.split()
                    if words:
                        if words[0] == "ping" and len(words) == 4:
                            self.on_ping(addr, *words[1:])
                        elif words[0] == "ack":
                            self.on_ack(addr, int(words[1]))

                except:
                    raise
                    continue
        finally:
            purger.stop()
            purger.join()
        
    @synchronized                
    def on_ping(self, ping_addr, port, tag, h):
        print ("on_ping(%s, %s, %s, %s)" % (ping_addr, port, tag, h))
        port = int(port)
        if False and h != "%x" % (hash((port, tag)),):
            print ("Invalid message %s %s %s from %s - hash mismatch" % (port, tag, h, ping_addr))
            return
        ip, ping_port = ping_addr
        worker_addr = (ip, port)
        if not tag in self.Registry:    self.Registry[tag] = {}
        worker_id = (ip, ping_port, port)
        if worker_id in self.Registry[tag]:
            wi = self.Registry[tag][worker_id]
            print("ping old worker")
            wi.ping()
        else:
                if worker_id in self.KnownWorkers:
                        wi = self.KnownWorkers[worker_id]
                else:
                        wi = self.KnownWorkers[worker_id] = WorkerInfo(worker_addr, tag)
                self.Sock.sendto("key %s" % (wi.Key,), ping_addr)
                print("ping new worker")

    @synchronized                
    def on_ack(self, ping_addr, work_port):
        ip, ping_port = ping_addr
        worker_id = (ip, ping_port, work_port)
        if worker_id in self.KnownWorkers:
                wi = self.KnownWorkers[worker_id]
                tag = wi.Tag
                if not tag in self.Registry:
                        self.Registry[tag] = {}
                registry = self.Registry[tag]
                registry[worker_id] = wi
        
    @synchronized
    def getList(self, tags, salt, n = None):
        tag = None if tags is None else tags[0] # for now, only 1 tag, ignore the rest
        if tags is None:
            # all workers
            lst = [
                    wi
                        for dct in self.Registry.values()
                        for wi in dct.values()
                ]        
        else:
            lst = self.Registry.get(tag, []).values()
        if not lst: return []
        servers = sorted(lst, key=lambda wi: wi.Addr)           
        by_addr = {}
        for wi in servers:
                ip = wi.Addr[0]
                lst = by_addr.setdefault(ip, [])
                lst.append(wi)
        #
        # sort by index within node and then by node
        #
        n_ips = len(by_addr)
        lst2 = []
        for i_ip, lst in enumerate(by_addr.values()):
                for j, wi in enumerate(lst):
                        lst2.append((j*n_ips+i_ip, wi))
        lst = [wi for inx, wi in sorted(lst2)]
        if n is not None:
                lst = lst[:n]
        return lst

        # the rest never worked
                
        if n is not None and len(servers) > n:
            state = random.getstate()
            random.seed(hash(salt))
            servers = random.sample(servers, n)
            random.setstate(state)
        return servers
        
    @synchronized
    def workersInfo(self):
        return [wi
                for dct in self.Registry.values()
                for wi in dct.values()
        ]
        
    @synchronized
    def workerInfo(self, addr):
        for dct in self.Registry.values():
            if addr in dct:
                wi = dct[addr]
                return wi
        return None
        
    def workers(self, sid=0, tags=None):
        return self.getList(tags, sid)
        
        
if __name__ == "__main__":
    import sys, getopt
    from wsgi_py import HTTPServer
    
    Usage = """python registry.py \
                -l <listener UDP port (default:7867)> 
                -w <web service port (default: 9867)>
    """
    
    listener_port = 7867
    http_port = 9867
    
    opts, args = getopt.getopt(sys.argv[1:], "l:w:")
    for opt, val in opts:
        if opt == '-l': listener_port = int(val)
        if opt == '-w': http_port = int(val)
        
    print("Listener UDP port: ", listener_port)
    print("Web service port:  ", http_port)

    registry = Registry()
    ping_receiver = PingReceiver(registry, listener_port)

    http_server = HTTPServer(http_port, WebServiceApp(registry), "*")
    
    registry.start()
    ping_receiver.start()
    http_server.start()

    registry.join()
    ping_receiver.join()
    http_server.join()
    
    
