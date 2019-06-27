from threading import RLock, Thread, Event, Condition

Waiting = []
In = []

def synchronized(method):
    def smethod(self, *params, **args):
        #print "@synchronized: wait %s..." % (method,)
        q = "%s(%x).%s" % (self, id(self), method)
        Waiting.append(q)
        with self:
            Waiting.remove(q)
            #print "@synchronized: in %s" % (method,)
            In.append(q)
            out = method(self, *params, **args)
        #print "@synchronized: out %s" % (method,)
        In.remove(q)
        return out
    return smethod

def printWaiting():
    print ("waiting:----")
    for w in Waiting:
        print (w)
    print ("in:---------")
    for w in In:
        print (w)

class Lockable:
    def __init__(self):
        self._Lock = RLock()
        self._WakeUp = Condition(self._Lock)

    def __enter__(self):
        return self._Lock.__enter__()
        
    def __exit__(self, exc_type, exc_value, traceback):
        return self._Lock.__exit__(exc_type, exc_value, traceback)

    def wait(self, timeout = None):
        with self._Lock:
            self._WakeUp.wait(timeout)
        
    def notify(self, n=1):
        with self._Lock:
            self._WakeUp.notify(n)
        

class MyThread(Thread, Lockable):
    def __init__(self):
        Thread.__init__(self)
        Lockable.__init__(self)

class Queue(Lockable):

    def __init__(self, capacity=None):
        Lockable.__init__(self)
        self.Capacity = capacity
        self.List = []
        self.Closed = False
    
    @synchronized
    def close(self):
        self.Closed = True
        self.notify()
        
    @synchronized    
    def add(self, item):
        assert not self.Closed, "The queue is closed"
        while self.Capacity is not None and len(self.List) >= self.Capacity:
            self.wait()
        self.List.append(item)
        self.notify()
        
    @synchronized    
    def push(self, item):
        assert not self.Closed, "The queue is closed"
        while self.Capacity is not None and len(self.List) >= self.Capacity:
            self.wait()
        self.List.insert(0, item)
        self.notify()

    @synchronized
    def pop(self):
        while len(self.List) == 0 and not self.Closed:
            self.wait()
        if len(self.List) == 0:
            return None     # closed
        item = self.List[0]
        self.List = self.List[1:]
        self.notify()
        return item
        
    @synchronized
    def flush(self):
        self.List = []
        self.notify()

    @synchronized
    def __len__(self):
        return len(self.List)
        
        
        
