from striped.pythreader import PyThread
from socket import socket, AF_INET, SOCK_DGRAM, gethostbyname
import json, select, sys, time
from .DataExchange2 import to_str, to_bytes

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

if PY2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

class WorkerRegistryClient(object):
    
    def __init__(self, registry_url):
        self.URL = registry_url
        
    def workers(self, sid="0", tag="default", n=None):
        url = "%s/list?sid=%s&tag=%s" % (self.URL, sid, tag)
        if n is not None:   url += "&n=%d" % (n,)
        response = urlopen(url)
        out = []
        if response.getcode()//100 == 2:
            out = [tuple(x) for x in json.loads(response.read())]
        return out
        
class WorkerRegistryPinger(PyThread):

    def __init__(self, registry_address, port, tag, ping_interval=3):
        PyThread.__init__(self)
        self.RegistryAddress = (gethostbyname(registry_address[0]), registry_address[1])
        self.Port = port
        self.Tag = tag
        self.PingInterval = ping_interval
        self.Stop = False
        self.Hash = "%x" % (hash((self.Port, self.Tag)),)
        self.Key = None
        
    def run(self):
        sock = socket(AF_INET, SOCK_DGRAM)
        sock_fd = sock.fileno()
        poll = select.poll()
        poll.register(sock_fd, select.POLLIN)
        while not self.Stop:
            ping = ("ping %s %s %s" % (self.Port, self.Tag, self.Hash)).encode("utf-8")
            sock.sendto(ping, self.RegistryAddress)
            #print "ping sent"
            t0 = time.time()
            t1 = t0 + self.PingInterval
            events = poll.poll(self.PingInterval*1000)
            #print "return from poll()"
            for fd, event in events:
                #print "poll event for fd=",fd
                if fd == sock_fd:
                    try:    
                        msg, addr = sock.recvfrom(10000)
                        #print("WorkerRegistryPinger.run: msg %s from %s" % (msg, addr))
                    except: 
                        pass
                    else:
                        #print "got message from", addr, msg
                        if addr == self.RegistryAddress:                # accept keys only from the registry
                            msg = to_str(msg)
                            words = msg.split()
                            if len(words) == 2 and words[0] == "key":
                                #print "received key: ", words[1]
                                self.Key = words[1]
                                #print "Got key %s" % (self.Key,)
                                sock.sendto(to_bytes("ack %d" % (self.Port,)), addr)
                                #print("WorkerRegistryPinger.run: ack sent to %s" % (addr,))
            t = time.time()
            if t < t1:  time.sleep(t1-t)
            
    def stop(self):
        self.Stop = True

