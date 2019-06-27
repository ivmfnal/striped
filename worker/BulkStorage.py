import numpy as np, posix_ipc, mmap

#
# Stores a dictionary { "key" -> (ndarray or int or float or string) } in a POSIX IPC shared memory segment
#

class BulkStorage(object):
    def __init__(self, name, shm = None, data = {}):
        self.Name = name
        self.Shm = shm if shm is not None else posix_ipc.SharedMemory(self.Name)
        self.Size = self.Shm.size
        self.MM = mmap.mmap(self.Shm.fd, self.Size) if self.Size > 0 else None
        self.Data = { k:v for k,v in data.items() }
        self.Keys = self.Data.keys()
        self.Map = {}
        self.I = 0
        self.DataOffset = None
        
    @staticmethod
    def open(name):
        s = BulkStorage(name)
        s.readMap()
        return s
    
    @staticmethod
    def create(name, data_dict):
        size = 0
        for k, v in data_dict.items():
            if isinstance(v, (str, bytes)):
                size += len(v) + 100
            elif isinstance(v, (int, float)):
                size += 100
            elif isinstance(v, np.ndarray):
                size += len(v.data) + 100
            size += len(k)
        shm = posix_ipc.SharedMemory(name, size=size, flags=posix_ipc.O_CREAT)
        s = BulkStorage(name, shm, data_dict)
        s.save()
        return s
    
    @staticmethod
    def delete(name):
        try:
            shm = posix_ipc.SharedMemory(name)
            shm.unlink()
        except:
            pass

    def __getitem__(self, key):
        if key in self.Data:
            return self.Data[key]
        if not key in self.Map:
            raise KeyError(key)
        #
        # If the value is scalar, it would have been in the Data already. So here we deal with arrays only
        #
        tup = self.Map[key]
        typ, info = tup[0], tup[1:]
        if typ == 'a':
            off, size, dtype, shape = info
            off = self.DataOffset + off
            value = np.frombuffer(self.MM[off:off+size], dtype=dtype).reshape(shape)
            self.Data[key] = value
        elif typ in 'sb':
            off, size = info
            off = self.DataOffset + off
            value = self.MM[off:off+size]
            self.Data[key] = value
        else:
            raise ValueError("Unknown value type %s for key %s" % (type(value), key))
        return value
        
    def unlink(self):
        try:
            self.Shm.close_fd()
            self.Shm.unlink()
        except:
            pass

    def initMap(self):
        self.Map = {}
        self.MM.seek(0)
        self.MM.write("\n")
        self.DataOffset = 1
        self.MM.seek(0)
        self.MM.flush()
        
    def save(self):
        # this is used only when we create the storage from a dictionary, so everything is in Data
        if self.MM is not None:
            # only if the data is not empty
            i = j = 0
            self.MM.seek(0)
            data = []
            for key, value in self.Data.items():
                if isinstance(value, str):
                    header = "%s:s:%d:%d\n" % (key, j, len(value))
                    data.append(value)
                    j += len(value)
                elif isinstance(value, bytes):
                    header = "%s:b:%d:%d\n" % (key, j, len(value))
                    data.append(value)
                    j += len(value)
                elif isinstance(value, (int, float)):
                    header = "%s:n:%s\n" % (key, value)
                elif isinstance(value, np.ndarray):
                    size = len(value.data)
                    dtype = value.dtype.str
                    shape = value.shape
                    header = "%s:a:%d:%d:%s:%s\n" % (key, j, size, dtype, ",".join(["%d" % (dim,) for dim in shape]))
                    data.append(value.data)
                    j += size
                else:
                    raise ValueError("Unknown value type %s for key %s" % (type(value), key))
                self.MM.write(header)
            self.MM.write("\n")
            for b in data:
                self.MM.write(bytes(b))
            self.MM.flush()
        
    def readMap(self):
        m = {}
        data = {}
        self.DataOffset = 0
        if self.MM is not None:
            done = False
            self.MM.seek(0)
            while not done:
                line = self.MM.readline()
                line = line.strip()
                if not line:  break
                words = line.split(":")
                key, typ = words[:2]
                value = None
                if typ == 'n':
                    try:    value=int(words[2])
                    except: value=float(words[2])
                    data[key] = value
                elif typ in 'sb':
                    m[key] = (typ, int(words[2]), int(words[3]))
                elif typ == 'a':
                    offset, size, dtype, shape = words[2:]
                    shape = shape.split(",")
                    offset, size = int(offset), int(size)
                    #print int
                    shape = map(int, shape)
                    m[key] = ('a', offset, size, dtype, shape)
            self.DataOffset = self.MM.tell()
        self.Map = m
        self.Data = data
        
    def asDict(self):
        return dict(self.items())
        
    def keys(self):
        return list(set(self.Map.keys()) | set(self.Data.keys()))
        
    def items(self):
        return [(k, self[k]) for k in self.keys()]
    
if __name__ == "__main__":
        data = {
                "a":    np.random.random((3,3)),
                "s":    "hello\n world",
                "i":    314,
                "f":    3.1415,
                "b":    bytes("hello there")
        }
        s = BulkStorage.create("test", data)

        s1 = BulkStorage.open("test")
        print s1.keys()
        for k, v in s1.items():
                print k, type(v), v
        s1.unlink()
