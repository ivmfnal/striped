import socket, zlib, traceback
import numpy as np

EOFException = Exception()

def to_str(x):
        if isinstance(x, str):  return x
        elif isinstance(x, bytes):      return x.decode("utf-8")
        else:   raise ValueError("Unknown input data type %s" % (type(x),))

def to_bytes(x):
        if isinstance(x, bytes):        return x
        elif isinstance(x, str):        return x.encode("utf-8", "ignore")
        else:   return bytes(x)

class DXMessage:

    Del = b' '     # defautlt delimiter
    Sync = Del + Del + Del

    def __init__(self, typ, **args):
        self.Type = typ
        self.Args = args
        
    def __str__(self):
        s = "DXMessage(del=%s, type='%s'" % (repr(self.Del), self.Type)
        for n, d in self.Args.items():
            if isinstance(d, (int, float, bool)) or d is None:
                s += " %s=%s" % (n, d)
            elif isinstance(d, str) and len(s) < 20:
                s += " %s=%s" % (n, repr(d))
            elif isinstance(d, (str, bytes)):
                s += " %s=%s(%d)" % (n, type(d), len(d))
            else:
                s += " %s=%s" % (n, repr(d))
        s += ')'
        return s
        
    __repr__ = __str__

    def __setitem__(self, name, value):
        self.Args[name] = value
        
    def append(self, *params, **kv):
        #
        # append(name, value, name, value, ...)
        # append(name=value, name=value...)
        #
        d = {}
        for i in range(0, len(params), 2):
            d[params[i]] = params[i+1]
        d.update(kv)
        for k, v in d.items():
                assert isinstance(v, (int, float, str, bool, type(None), np.ndarray, bytes)), "Unknown value type: %s" % (type(v),)
        self.Args.update(d)
        return self

    __call__ = append
    
    def __getitem__(self, name):
        return self.Args[name]
            
    def get(self, key, default=None):
        try:    value = self[key]
        except KeyError:
                value = default
        return value
            
    def __contains__(self, name):
        return name in self.Args
        
    def keys(self):
        return self.Args.keys()
        
    def items(self):
        return self.Args.items()
        
    def serialize(self):
        msg = [to_bytes(self.Type)]
        for n, v in self.Args.items():
                if isinstance(v, (bytes, str)):
                        t = b'b' if isinstance(v, bytes) else b's'
                        msg.append(b"%s=%s%d:%s" % (to_bytes(n), t, len(v), to_bytes(v)))
                elif isinstance(v, np.ndarray):
                        data = bytes(np.ascontiguousarray(v).data)
                        msg.append(to_bytes("%s=n%s:%s:%d:" % (n, v.dtype.str, 
                                ",".join(["%d" % (x,) for x in v.shape]), len(data))) + data)
                else:
                        assert isinstance(v, (int, float, bool)) or v is None, "Unknown DXMessage argument type %s: %s" % (type(v), repr(v))
                        msg.append(to_bytes("%s==%s" % (n, v)))
        msg = self.Del.join(msg) + self.Del
        return b"%s%d:%s" % (self.Sync, len(msg), msg)

    @staticmethod
    def fromBuffered(buffered):
        def decode_value(b):
                if b in (b'True', b'False'):    return b == b'True'
                elif b == b'None':              return None
                else:
                        try:    x = int(b)
                        except: x = float(b)
                        return x


        sync = buffered.readn(3)
        if not sync:	
            #print("DXSocket.fromBuffered(): EOF")
            return None		# EOF
        #print("DXSocket.fromBuffered(): sync received")

        if sync != DXMessage.Sync:
                raise IOError("Stream is out of sync. Received %s instead of sync %s" % (to_str(sync), to_str(DXMessage.Sync)))

        msg_size = buffered.readuntil(b":")
        if not msg_size:
                raise IOError("Incomplete message. Sync received, but can not read message size")

        msg_size = int(msg_size)
        body = buffered.readn(msg_size)
        if len(body) < msg_size:
                raise IOError("Incomplete message. Expected %d bytes, received %d" % (msg_size, len(body)))

        buf = BufferedBytes(body)
        msg_type = to_str(buf.readuntil(DXMessage.Del))
        message = DXMessage(msg_type)
        while not buf.isEmpty():
                name = to_str(buf.readuntil(b'='))
                #print ("name=", name)
                if not name:
                        raise ValueError("Emty field name")
                t = buf.readn(1)
                if t == b'=':
                        value = decode_value(buf.readuntil(DXMessage.Del))
                elif t == b'n':
                        dtype = buf.readuntil(b':')
                        shape = buf.readuntil(b':')
                        shape = tuple([int(x) for x in shape.split(b',')])
                        size = int(buf.readuntil(b':'))
                        data = buf.readn(size)
                        value = np.frombuffer(data, dtype=dtype).reshape(shape)
                        d = buf.readn(1) 
                        assert d == DXMessage.Del, "Expeted to see delimiter %s after the field, got %s" % (repr(DXMessage.Del), repr(d))
                elif t in (b's', b'b'):
                        l = int(buf.readuntil(b':'))
                        value = buf.readn(l)
                        #print ("value from buffer:", value)
                        if t == b's':   value = to_str(value)
                        d = buf.readn(1) 
                        assert d == DXMessage.Del, "Expeted to see delimiter %s after the field, got %s" % (repr(DXMessage.Del), repr(d))
                message[name] = value
                #print("name: %s" % (repr(value),))
        return message
                
        
    def toSocket(self, sock):
        sock.send(self.serialize())
        
    def toConnection(self, conn):       # send to the multiprocessing.Connection
        conn.send(self.serialize())
        
    def toFile(self, f):
        f.write(self.serialize())
       
    @staticmethod
    def fromBufferedSocket(sock):
        return DXMessage.fromBuffered(sock)
            
    def toExchange(self, exchange):
        exchange.send(self)
        
class Context(object):
    
    def __enter__(self):
        return self
        
    def __exit__(self, *params):
        self.close()

class BufferedSocket(Context):

    def __init__(self, sock):
        self.Sock = sock
        self.Buffer = b''

    def readuntil(self, stop):
        #print ("readuntil(%s)..." % (repr(stop),))
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        while not stop in word:
                word = self.Sock.recv(1000000)
                if not word:    raise IOError("Socket closed while reading until %s" % (repr(stop),))
                else:           buflst.append(word)
        buf = b''.join(buflst)
        head, self.Buffer = buf.split(stop,1)
        #print ("readuntil(%x): <%s>_<%s>" % (ord(stop), head, self.Buffer))
        return head
                
    def readn(self, n):
        #print ("readn(%d)..." % (n,))
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        length = len(word)
        while length < n:
                word = self.Sock.recv(1000000)
                if not word:    
                    #print("readn: eof empty recv")
                    break   # eof
                length += len(word)
                buflst.append(word)
        data = b''.join(buflst)
        out, self.Buffer = data[:n], data[n:]
        #print ("readn(%d): <%s>_<%s>" % (n, out, self.Buffer))
        return out

    def flush(self):
        w = self.Buffer
        self.Buffer = b''
        return w
            
    def close(self):
        try:    self.Sock.close()
        except: pass

class BufferedBytes(Context):

        def __init__(self, buf):
                self.Buffer = buf
                self.I = 0

        def close(self):
                pass

        def readuntil(self, stop):
                i = self.Buffer.find(stop, self.I)
                if i >= 0:
                        new_i = i + len(stop)
                        out = self.Buffer[self.I:i]
                        self.I = new_i
                        return out
                else:
                        raise IOError("End of buffer while reading until %s" % (repr(stop),))

        def readn(self, n):
                #print ("readn: I=%d, n=%d, L=%d" % (self.I, n, len(self.Buffer)))
                if self.I + n > len(self.Buffer):
                        raise IOError("End of buffer while reading %d bytes (buffer remaining: %d)" % (n, len(self.Buffer) - self.I))
                new_i = self.I + n
                out = self.Buffer[self.I:new_i]
                self.I = new_i
                return out

        def isEmpty(self):
                return self.I >= len(self.Buffer)

        def flush(self):
                self.Buffer, self.I = b'', 0
                        
                
        
            
class BufferedConnection(Context):

    #
    # conn in multiprocessing.Connection object
    #
    def __init__(self, conn):
        self.Conn = conn
        self.Buffer = b''

    def close(self):
        try:    self.Conn.close()
        except: pass
        
    def readuntil(self, stop):
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        while not stop in word:
                try:    word = self.Conn.recv_bytes()
                except EOFError:
                        word = ''
                if not word:    break
                else:           buflst.append(word)
        buf = b''.join(buflst)
        if not stop in buf:
                return None             # eof
        head, self.Buffer = buf.split(stop,1)
        return head
                
    def readn(self, n):
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        length = len(word)
        while length < n:
                try:    word = self.Conn.recv_bytes()
                except EOFError:
                        word = b''
                if not word:    break   # eof
                length += len(word)
                buflst.append(word)
        if length < n:
                return None     # EOF
        data = b''.join(buflst)
        out, self.Buffer = data[:n], data[n:]
        return out

    def flush(self):
        w = self.Buffer
        self.Buffer = b''
        return w

class DataExchangeSocket(BufferedSocket):

    def __init__(self, sock, address = None):
        BufferedSocket.__init__(self, sock)
        self.Address = address
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10000000)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10000000)
        
    @staticmethod
    def connect(address):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(address)
        return DataExchangeSocket(sock, address)

    def accept(self):
        s, addr = self.Sock.accept()
        return DataExchangeSocket(s, addr)

    def recv(self):
        try:    msg = DXMessage.fromBufferedSocket(self)
        except: 
            #print ("DataExchangeSocket.recv() error: %s" % (traceback.format_exc(),))
            msg = None
        return msg
        
    def send(self, msg):
        msg.toSocket(self.Sock)
        
    def __del__(self):
        try:    self.Sock.close()
        except: pass
                        
if __name__ == '__main__':
    a=np.random.random((2,2))
    msg = DXMessage("hello", arr=a, barg=b"xyz", sarg="zyx")(A="value of A\n is here")(BB = "a b c d").append(bl=True, i=3, f=3.14, n=None)
    serialized = msg.serialize()
    print ("serialized:", serialized)
    msg1 = DXMessage.fromBuffered(BufferedBytes(serialized))

    print (msg)
    print (msg1)
    
    
    
