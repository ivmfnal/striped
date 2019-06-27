from striped.pythreader import Primitive, PyThread, Task, TaskQueue, synchronized
#import posix_ipc, mmap 
import socket, time
import threading
from .dataEncoder import encodeData


_BUFSIZE = 10000000

class Forwarder(PyThread):

    def __init__(self, name, data, port, forward_to):
        PyThread.__init__(self)
        self.Name = name
        self.Data = data
        self.Server = forward_to[0]
        self.ForwardTo = forward_to[1:]
        self.Port = port
        self.Response = None
        
    def run(self):
        t0 = time.time()
        #print "Forwarder:%s: forwarding to %s -> %s" % (threading.current_thread().name, self.Server, self.ForwardTo)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _BUFSIZE)
        sock.connect((self.Server, self.Port))
        
        header = "%s %d" % (self.Name, len(self.Data))
        if self.ForwardTo:
            for a in self.ForwardTo:
                header += " " + a
        header += '\n'
        
        sock.send(header)
        sock.send(self.Data)
        self.Response = sock.recv(10)
        #print "Forwarder:%s: done: [%s] time=%s" % (threading.current_thread().name, self.Response, time.time() - t0)
        
class BulkDataSender(object):
    
    NSplit = 3
        
    def __init__(self, name, data, port, addresses):
        #print "BulkDataSender(%s, %s, %s, %s)" % (name, len(data), port, addresses)
        self.Port = port
        self.Addresses = addresses
        self.Forwarders = []
        self.Name = name
        if not isinstance(data, (str, bytes)):
            data = encodeData(data)
        self.Data = data
        
    def start(self):
        n = len(self.Addresses)
        if n > 0:
            nsegment = (n + self.NSplit - 1)/self.NSplit
            forwarders = []
            for i in range(0, n, nsegment):
                lst = self.Addresses[i:i+nsegment]
                if lst:
                    f = Forwarder(self.Name, self.Data, self.Port, lst)
                    f.start()
                    forwarders.append(f)
            self.Forwarders = forwarders
        
    def wait(self):
        for f in self.Forwarders:
            #print "joining %s..." % (f.name,)
            f.join()
            assert f.Response == "OK"
        
class Handler(Task):

    def __init__(self, transport, sport, sock):
        Task.__init__(self)
        self.Sock = sock
        self.SPort = sport
        self.Transport = transport
        
    def run(self):
        try:
                self.Sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _BUFSIZE)
                header = ""
                while not "\n" in header:
                    text = self.Sock.recv(1000)
                    if not text:
                        return      # premature EOF
                    header += text
                header, data = header.split("\n", 1)
                words = header.split()
                name, size = words[:2]
                forward_to = words[2:]
                size = int(size)
                data_len = len(data)
                data = [data]
                while data_len < size:
                    part = self.Sock.recv(size - data_len)
                    if not part:
                        return      # premature EOF
                    data_len += len(part)
                    data.append(part)
                data = ''.join(data)
                
                #print "Handler: received data [%s], %d bytes" % (name, len(data))
                #print "Handler: forwarding to: %s" % (forward_to,)
                client = BulkDataSender(name, data, self.SPort, forward_to)
                client.start()

                self.Transport.add(name, data)

                # save data into shm

                """
                
                shm = posix_ipc.SharedMemory(name, size=size, flags=posix_ipc.O_CREAT)
                mm = mmap.mmap(shm.fd, size)
                mm.seek(0)
                mm.write(data)
                mm.flush()

                """
                
                # wait for forwarders
                
                client.wait()
                    
                self.Sock.send("OK")
        finally:
                # no matter what happens, clean up
                self.Transport = None
                self.Sock.close()
        
class BulkDataTransport(PyThread):

    def __init__(self, port):
        PyThread.__init__(self)
        self.Port = port
        self.Stop = False
        self.TaskQueue = TaskQueue(10, capacity=10)
        self.Data = {}

    @synchronized
    def add(self, name, data):
        self.Data[name] = data
        #print "BulkDataTransport: adding [%s]" % (name,)
        self.wakeup()

    @synchronized
    def pop(self, name, timeout = None):
        t0 = time.time()
        t1 = None if timeout is None else t0 + timeout
        while not name in self.Data and (timeout is None or time.time() < t1):
            if timeout is not None:
                self.sleep(max(0, t1-time.time()))
            else:
                self.sleep()
        #print "BulkDataTransport: pop: keys: %s" % (self.Data.keys(),)
        data = self.Data.pop(name)
        return data
        
    def run(self):
        ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ssock.bind(('', self.Port))
        ssock.listen(5)
        
        while not self.Stop:
            sock, addr = ssock.accept()
            task = Handler(self, self.Port, sock)
            self.TaskQueue << task
            
    def stop(self):
        self.Stop = True

def send_bulk_data(name, data, port, addresses):
    client = BulkDataSender(name, data, port, addresses)
    client.start()
    client.wait()


if __name__ == "__main__":

    import sys
    
    port = 1234
    
    if sys.argv[1] == "server":
        repeater = BulkDataTransport(port)
        repeater.start()
        repeater.join()
    else:
        # client
        print send_to(sys.argv[2], open(sys.argv[3], "rb").read(), port, sys.argv[4:])
    
            
            
