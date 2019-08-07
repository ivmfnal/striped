import socket, zlib, traceback

class DXMessage:

    D = '`'     # defautlt delimiter

    def __init__(self, typ, *params, **args):
        self.Type = typ
        self.Params = params
        self.Args = args
        self.Body = {}
        self.Del = self.D
        
    def delimiter(self, d):
        self.Del = d
        return self
        
    def __str__(self):
        s = "DXMessage(del='%s', type='%s', %s, %s" % (self.Del, self.Type, self.Params, self.Args)
        for n, d in self.Body.items():
            s += ", %s(%d)" % (n, len(d))
        s += ')'
        return s
        
    __repr__ = __str__

    def __setitem__(self, name, value):
        self.Body[name] = value
        
    def append(self, *params, **dct):
        #
        # append(name, value)
        # append(name=value, name=value...)
        #
        if params:
            self.Body[params[0]] = params[1]
        else:
            self.Body.update(dct)
        return self

    __call__ = append
    
    def __getitem__(self, name):
        if isinstance(name, int):
            return self.Params[name]
        if name in self.Args:
            return self.Args[name]
        else:
            return self.Body[name]
            
    def get(self, key, default=None):
        try:    value = self[key]
        except KeyError:
                value = default
        return value
            
    def __contains__(self, name):
        return name in self.Body
        
    def keys(self):
        return self.Body.keys()
        
    def items(self):
        return self.Body.items()
        
    def serialize(self):
        for k, v in self.Args.items():
            if not isinstance(v, (str, int, float, bool)) and not v is None:
                raise TypeError("DXMessage argument must be either string, float, bool, None, int. %s is %s" % (k, type(v)))
        for v in self.Params:
            if not isinstance(v, (str, int, float, bool)) and not v is None:
                raise TypeError("DXMessage param must be either string, float, bool, None, int. Got %s" % (type(v),))
        header = [self.Type] + \
            ["%s" % (p,) for p in self.Params] + \
            ["%s=%s" % (k, v) for k, v in self.Args.items()]
        header = self.Del + self.Del + self.Del + self.Del.join(header) + '\n'     # repeat the delimiter 3 times as the sync mark
        yield header.encode("utf-8")
        for n, b in self.Body.items():
            if b is not None:
                #print "serialize: header=[%s]" % (header,)
                #print "DataExchange.serialize:%s=%s" % (n, b)
                original_size = len(b)
                header = "+%s%s%s%d\n" % (self.Del, n, self.Del, original_size)
                if False and len(b) > 1000:         # never compress
                    # compress body
                    original_size = len(b)
                    compressed = zlib.compress(b)
                    compressed_size = len(compressed)
                    if compressed_size < original_size:
                        header = "+%s%s%s%d%sz%s%d\n" % (self.Del, n, self.Del, compressed_size, self.Del, self.Del, original_size)
                        b = compressed
                yield header.encode("utf-8")
                if isinstance(b, str):
                    b = b.encode("utf-8")
                yield b
        yield "\n".encode("utf-8")
        
    def toSocket(self, sock):
        parts = list(self.serialize())
        #print("toSocket: parts:")
        #for p in parts:
        #    print (type(p), p)
        serialized = b''.join(parts)
        sock.send(serialized)
        
    def toConnection(self, conn):       # send to the multiprocessing.Connection
        for x in self.serialize():
            conn.send_bytes(x)
        
    def toFile(self, f):
        for x in self.serialize():
            f.write(x)
       
    @staticmethod
    def fromBuffered(buffered):
        def cvt(text):
            if text in ("None", b"None"):
                return None
            try:    value = int(text)
            except:
                try:    value = float(text)
                except:
                    value = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
            print("fromBuffered.cvt(%s) -> %s" % (repr(text), repr(value)))
            return value

        print("fromBuffered: read SYNC...")
        try:
             sync = buffered.readn(3)
        except:
             print (traceback.format_exc())
        if not sync:
                return None             # EOF

        assert len(sync) == 3 and sync[1] == sync[0] and sync[2] == sync[0], \
                "DataExchange stream is out of sync. SYNC:%s" % (repr(sync),)

        delimiter = sync[:1]
        print("fromBuffered: sync received: %s, delimiter: %s" % (repr(sync), repr(delimiter)))
         
        header = buffered.readuntil(b'\n')
        if not header:
            return None     # EOF ?
        
        # parse header
        print("fromBuffered: header: %s, delimiter: %s" % (repr(header), repr(delimiter)))
        words = header.split(delimiter)     # first word is always empty
        #print "DXMessage: header:", words
        # check sync
        if len(words) < 1:
            raise RuntimeError("Can not parse message header: [%s] delimiter:'%s'" % (header, delimiter))
        command = words[0].decode("utf-8", "ignore")
        params = []
        args = {}
        for w in words[1:]:
            print("fromBuffered: word:", repr(w))
            pair = w.split(b"=", 1)
            if len(pair) == 2:
                args[pair[0].decode("utf-8", "ignore")] = cvt(pair[1])
            else:
                params.append(cvt(pair[0]))     
        
        out = DXMessage(command, *params, **args)
        done = False
        #print "Header parsed: %s" % (out,)
        while not done:
                h = buffered.readn(1)
                if not h:
                        return None
                elif h == b'+':
                            # read next part
                            part_hdr = buffered.readuntil(b"\n")
                            #print "part header:", part_hdr
                            if not part_hdr:    return None
                            #
                            # part header format:
                            # <empty><d><name><d><compressed size>
                            # <empty><d><name><d><compressed size><d><options>
                            # if the part is compressed, 'z' is in options and format:
                            # <empty><d><name><d><compressed size><d><options><d><original size>
                            #
                            words = part_hdr.split(delimiter)
                            if len(words) < 3 or words[0] != b'':
                                raise ValueError("Can not parse attachment header: [%s] (must start with the delimiter '%s'" % (body_hdr,delimiter))
                            name, size = words[1:3]
                            original_size = size
                            opts = b""
                            if len(words) > 3:
                                opts = words[3]
                                if b'z' in opts:
                                    original_size = int(words[4])
                            size = int(size)
                            data = buffered.readn(size)
                            if size > 0 and not data: return None            # eof
                            assert len(data) == size, "Received incomplete attachment. Expected length:%d, received:%d" % (size, len(data)) 
                            if b'z' in opts:
                                data = zlib.decompress(data)
                                assert len(data) == original_size, \
                                        "Decompressed size of the attachment is incorrect. Expected length:%d, received:%d" % (original_size, len(data)) 
                            out[name.decode("utf-8","ignore")] = data
                            #print "DXMessage: part <%s> received" % (name,)
                elif h == b'\n':
                        # end of message
                        print("fromBuffered: end of message")
                        done = True
                else:
                        raise ValueError("Received %s instead of part begin '+' or end-of-message '\\n'" % (repr(h),))
        print("fromBuffered: received message: %s" % (out,))
        return out

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
        print ("readuntil(%s)..." % (repr(stop),))
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        while not stop in word:
                word = self.Sock.recv(1000000)
                if not word:    break
                else:           buflst.append(word)
        buf = b''.join(buflst)
        if not stop in buf:
                return None             # eof
        head, self.Buffer = buf.split(stop,1)
        print ("readuntil(%x): <%s>_<%s>" % (ord(stop), head, self.Buffer))
        return head
                
    def readn(self, n):
        print ("readn(%d)..." % (n,))
        word = self.Buffer
        self.Buffer = b''
        buflst = [word]
        length = len(word)
        while length < n:
                word = self.Sock.recv(1000000)
                if not word:    
                    print("readn: eof empty recv")
                    break   # eof
                length += len(word)
                buflst.append(word)
        if length < n:
                return None     # EOF
        data = b''.join(buflst)
        out, self.Buffer = data[:n], data[n:]
        print ("readn(%d): <%s>_<%s>" % (n, out, self.Buffer))
        return out

    def flush(self):
        w = self.Buffer
        self.Buffer = b''
        return w
            
    def close(self):
        try:    self.Sock.close()
        except: pass
        
            
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
            print ("DataExchangeSocket.recv() error: %s" % (traceback.format_exc(),))
            msg = None
        return msg
        
    def send(self, msg):
        msg.toSocket(self.Sock)
        
    def __del__(self):
        try:    self.Sock.close()
        except: pass
                        
if __name__ == '__main__':
    msg = DXMessage("hello", arg="xyz", arg3="zyx")(A="value of A\n is here")(BB = "a b c d")
    
    
    
    serialized = list(msg.serialize())
    print("serialzed: [%s]" % (''.join(serialized)))
    print(DXMessage.deserialize(msg.serialize()))
    print(DXMessage.deserialize(''.join(serialized)))
    
    
    
