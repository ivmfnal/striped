import json, base64, numpy as np, sys
from json import JSONEncoder

PY3 = sys.version_info >= (3,)

class DataEncoder(JSONEncoder):
    
    def default(self, o):
        if isinstance(o, np.ndarray):
            o = {
                "///ndarray///": 1.14159265,
                "dtype":o.dtype.str,
                "shape":o.shape,
                "data":base64.b64encode(np.ascontiguousarray(o).data)
            }
            return o
        elif isinstance(o, bytes):
           if PY3:
               return o.decode("utf-8")
           else:
               return str(o)
        else:
               return JSONEncoder.default(self, o)
    
    @staticmethod
    def objecthook(obj):
        if obj.get("///ndarray///") == 1.14159265:
            return np.frombuffer(base64.b64decode(obj["data"]), 
			obj["dtype"]).reshape(obj["shape"])
        else:
            out = {}
            for k, v in obj.items():
                if PY3:
                    if isinstance(k, bytes): k = k.decode("utf-8")
                    if isinstance(v, bytes): v = v.decode("utf-8")
                else:
                    #PY2
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
	data = {
		"a":	1,
		"b":	1.1,
		"c":	"hello",
		"d":	np.random.random((5,5))
	}
	encoded = encodeData(data)
	decoded = decodeData(encoded)
	print ("original:") 
	pprint.pprint(data)
	print ("encoded :") 
	pprint.pprint(encoded)
	print ("decoded :") 
	pprint.pprint(decoded)
