from striped.pythreader import Primitive, synchronized
import time, uuid, os

def random_bytes(n):
    return os.urandom(n)


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
        elif isinstance(value, long):
            self.U = uuid.UUID(int=value)
        else:
            raise ValueError("Unsupported type for source Token value: %s" % (type(value),))
        self.Data = None
            
    def minExpiration(self, texp):
        if isinstance(texp, Token):
            texp = texp.Expiration
        if texp is None:
            texp = self.Expiration
        elif self.Expiration is not None:
            texp = min(texp, self.Expiration)
        return texp
        
    def expired(self):
        return self.Expiration is not None and self.Expiration < time.time()
        
    def renew(self, ttl):
        self.Expiration = time.time() + ttl
        
    def encrypt(self, s):
        sbytes = bytes(s)
        key = self.bytes
        key_len = len(key)
        out = [chr(ord(sb) ^ ord(key[i % key_len])) for i, sb in enumerate(sbytes)]
        return bytes(''.join(out))

    def __xor__(self, another):
        if isinstance(another, Token):
            return Token(value = self.U.int ^ another.U.int, expiration = self.minExpiration(another))
        elif isinstance(another, (str, unicode)):
            return self.encrypt(another)
            
        
    @property
    def hex(self):
        return self.U.hex
        
    key = hex

    @property
    def int(self):
        return self.U.int
        
    @property
    def bytes(self):
        return self.U.bytes
        
class TokenStorage(Primitive):
    
    def __init__(self, ttl):
        Primitive.__init__(self)     
        self.Tokens = {}            # hex -> (token, data)
        self.TTL = ttl
    
    @synchronized
    def purge(self):
        # purge expired authenticators
        new_dict = {}
        t = time.time()
        for key, token in self.Tokens.items():
            if not token.expired():
                new_dict[key] = token
        self.Tokens = new_dict
        
    @synchronized    
    def token(self, key):
        self.purge()
        return self.Tokens.get(key)
        
    __getitem__ = token
        
    @synchronized    
    def tokens(self):
        self.purge()
        return self.Tokens.values()

    @synchronized    
    def createToken(self, data):
        self.purge()
        key = None
        while key is None or key in self.Tokens:
            token = Token(expiration=self.TTL)
            key = token.key
        token.Data = data
        self.Tokens[key] = token
        return token
