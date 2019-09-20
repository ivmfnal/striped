import json, base64, numpy as np, sys
from json import JSONEncoder

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

class DataEncoder(JSONEncoder):
    
    def default(self, o):
        if isinstance(o, np.ndarray):
            o = {
                "///type///": "ndarray",
                "dtype":o.dtype.str,
                "shape":o.shape,
                "data":base64.b64encode(np.ascontiguousarray(o).data)
            }
            return o
        elif isinstance(o, bytes):
           if PY3:
               o = {
                    "///type///": "bytes",
                    "data":base64.b64encode(o).decode("utf-8")
               }
               return o
           else:
               return str(o)
        else:
               return JSONEncoder.default(self, o)
    
    @staticmethod
    def objecthook(obj):
        typ = obj.get("///type///")
        if typ == "ndarray":
            return np.frombuffer(base64.b64decode(obj["data"]), 
			obj["dtype"]).reshape(obj["shape"])
        elif typ == "bytes":
            return base64.b64decode(obj["data"])
        else:
            out = {}
            for k, v in obj.items():
                if False and PY3:
                    print("objecthook:", repr(k), repr(v))
                    if isinstance(k, bytes): k = k.decode("utf-8")
                    if isinstance(v, bytes): v = v.decode("utf-8")
                elif PY2:
                    if isinstance(k, unicode): k = k.encode("utf-8", "ignore")
                    if isinstance(v, unicode): v = v.encode("utf-8", "ignore")
                out[k] = v
            return out

        
def encodeData(obj):
    return DataEncoder().encode(obj)

def decodeData(text):
    return json.loads(text, object_hook=DataEncoder.objecthook)

if __name__ == "__main__":
    import pprint
    import pickle
    data = {
            "a":    1,
            "b":    1.1,
            "c":    "hello",
            "d":    np.random.random((5,5)),
            "e":    pickle.dumps({"a":"b"})
    }
    encoded = encodeData(data)
    print ("encoded :") 
    print(type(encoded), encoded)
    
    decoded = decodeData(encoded)
    print ("original:") 
    pprint.pprint(data)
    print ("decoded :") 
    pprint.pprint(decoded)
    
    ev = pickle.loads(decoded["e"])
    print ("e value=", ev)
    
