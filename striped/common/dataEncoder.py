import json, base64, numpy as np
from json import JSONEncoder

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
        return JSONEncoder.default(self, o)
    
    @staticmethod
    def objecthook(obj):
        if obj.get("///ndarray///") == 1.14159265:
            return np.frombuffer(base64.b64decode(obj["data"]), 
			obj["dtype"]).reshape(obj["shape"])
        else:
            out = {}
            for k, v in obj.items():
                if isinstance(k, unicode): k = k.encode("utf-8", "ignore")
                if isinstance(v, unicode): v = v.encode("utf-8", "ignore")
                out[k] = v
            return out

        
def encodeData(obj):
    return DataEncoder().encode(obj)

def decodeData(text):
    return json.loads(text, object_hook=DataEncoder.objecthook)


