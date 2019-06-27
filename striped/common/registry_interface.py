from .MyThread import MyThread
from socket import socket, AF_INET, SOCK_DGRAM, gethostbyname
import json, select
try:
    from urllib2 import urlopen
except:
    from urllib.request import urlopen

class WorkerRegistryClient(object):
    
    def __init__(self, registry_url):
        self.URL = registry_url
        
    def workers(self, sid="0", tag="default", n=None):
        url = "%s/list?sid=%s&tag=%s" % (self.URL, sid, tag)
        if n is not None:   url += "&n=%d" % (n,)
        response = urlopen(url)
        out = []
        if response.getcode()/100 == 2:
            out = [tuple(x) for x in json.loads(response.read())]
        return out
        
class WorkerRegistryPinger(MyThread):

    def __init__(self, registry_address, port, tag, ping_interval=3):
        MyThread.__init__(self)
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
            sock.sendto("ping %s %s %s" % (self.Port, self.Tag, self.Hash), self.RegistryAddress)
            #print "ping sent"
            events = poll.poll(self.PingInterval*1000)
            #print "return from poll()"
            for fd, event in events:
                #print "poll event for fd=",fd
                if fd == sock_fd:
                    try:    
                        msg, addr = sock.recvfrom(10000)
                    except: 
                        pass
                    else:
                        #print "got message from", addr, msg
                        if addr == self.RegistryAddress:                # accept keys only from the registry
                            words = msg.split()
                            if len(words) == 2 and words[0] == "key":
				#print "received key: ", words[1]
                                self.Key = words[1]
                                #print "Got key %s" % (self.Key,)
                                sock.sendto("ack %d" % (self.Port,), addr)
            
    def stop(self):
        self.Stop = True

