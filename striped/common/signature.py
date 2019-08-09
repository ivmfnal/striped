import hashlib, base64, time, uuid, random, sys
from .DataExchange2 import to_bytes, to_str

PY3 = sys.version_info >= (3,)
PY2 = sys.version_info < (3,)

# init random state

rsave = random.getstate()
random.seed()               # init from random source
my_rstate = random.getstate()
random.setstate(rsave)

def hash_algorithm():
    prefered = ["sha256","sha512","md5"]
    alg = None
    for a in prefered:
        if a in hashlib.algorithms_guaranteed:
            alg = a
            break
    return alg

class Token(object):

    def __init__(self, value=None, expiration=None):
        if expiration is not None:
            if expiration < 365*24*3600:
                expiration = time.time() + expiration
        self.Expiration = expiration
        if value is None:
            self.U = uuid.UUID(int = uuid.uuid1().int ^ uuid.uuid4().int)
        elif isinstance(value, str):
            self.U = uuid.UUID(value)
        elif PY2 and isinstance(value, (long, int)) or PY3 and isinstance(value, int):
            self.U = uuid.UUID(int=value)
        else:
            raise ValueError("Unsupported type for source Token value: %s" % (type(value),))
            
    def minExpiration(self, texp):
        if isinstance(texp, Token):
            texp = texp.Expiration
        if texp is None:
            texp = self.Expiration
        elif self.Expiration is not None:
            texp = min(texp, self.Expiration)
        return texp
        
    def encrypt(self, s):
        sbytes = bytes(s)
        key = self.bytes
        key_len = len(key)
        out = [chr(ord(sb) ^ ord(key[i % key_len])) for i, sb in enumerate(sbytes)]
        return bytes(''.join(out))

    def __xor__(self, another):
        if isinstance(another, Token):
            return Token(value = self.U.int ^ another.U.int, expiration = self.minExpiration(another))
        elif PY3 and isinstance(another, str) or PY2 and isinstance(another, (str, unicode)):
            return self.encrypt(another)
            
        
    @property
    def hex(self):
        return self.U.hex

    @property
    def int(self):
        return self.U.int
        
    @property
    def bytes(self):
        return self.U.bytes
        
    
def random_salt():
    return Token().hex



class Signer(object):

	def __init__(self, key):
		self.Key = key
	
	def calculate(self, t, salt, data, algorithm):
	    if not algorithm in hashlib.algorithms_available:
	        raise ValueError("Unsupported hash algorithm '%s'" % (algorithm,))
	    h = hashlib.new(algorithm)
	    if isinstance(t, float):
	        t = int(t+0.5)
	    if isinstance(t, int):
	        t = "%d" % (t,)
	    s1 = "%s %s %s" % (t,salt,self.Key)
	    s2 = " ".join(map(str, data))
	    s = "%s %s" % (s1, s2)
	    h.update(to_bytes(s))
	    return h.hexdigest()

	def sign(self, data):
	    # choose hash algorithm
	    alg = hash_algorithm()
	    if not alg:
	        raise ValueError("None of prefered hash algorithms is supported")

	    # generate salt   
		 
	    salt = random_salt()
	    
	    # calculate the signature
	    
	    t = "%d" % (int(time.time()),)
	    sig = self.calculate(t, salt, data, alg)
	    #print "Signer.sign(%s, %s, %s, %d, %s) -> %s" % (self.Key, t, salt, len(data), alg, sig)
	    return sig, t, salt, alg

	def verify(self, client_signature, data, t, salt, alg, time_tolerance = 3600):
	    # t is supposed to be a string
	    now = int(time.time()+0.5)
	    t = int(t)
	    if abs(int(t) - now) > time_tolerance:
	        return False, "Time not synchronized: t:%d (%s), my time: %d (%s)" % (t, time.ctime(t), now, time.ctime(now))
	    my_signature = self.calculate(t, salt, data, alg)
	    #print "Signer.verify(%s, %s, %s, %d, %s) -> %s" % (self.Key, t, salt, len(data), alg, my_signature)
	    if my_signature == client_signature:
	        return True, "OK"
	    else:
	        return False, "Signature mismatch: expcted: %s, received: %s" % (my_signature, client_signature)
	    

	
	
    
