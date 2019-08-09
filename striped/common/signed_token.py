import hashlib, json, base64, time
from .rfc2617 import digest_client

def generate_secret(length):
    import os
    return os.urandom(length)


class SignedTokenSignatureVerificationError(Exception):
    pass
    
class SignedTokenHeaderError(Exception):
    pass
    
class SignedTokenExpiredError(Exception):
    pass
    
class SignedTokenImmatureError(Exception):
    pass
    
class SignedTokenUnacceptedAlgorithmError(Exception):
    def __init__(self, alg):
        self.Alg = alg
        
    def __str__(self):
        return "Unaccepted signature algorithm %s" % (self.Alg,)
        
class SignedTokenAuthoriztionError(Exception):
    def __init__(self, msg):
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8", "ignore")
        self.Message = msg
        
    def __str__(self):
        return "Authorization or authentication failed: %s" % (self.Message,)
        
class SignedToken(object):
    
    AcceptedAlgorithms = ["sha256","sha384","sha512","md5"]

    AvailableAlgorithms = set(hashlib.algorithms if hasattr(hashlib, "algorithms") else hashlib.algorithms_available)
    
    #PreferredAlgorithms = [a for a in AcceptedAlgorithms if a in AvailableAlgorithms]
    
    def __init__(self, payload, expiration=None, not_before=None):
        self.Alg = [a for a in self.AcceptedAlgorithms if a in self.AvailableAlgorithms][0]
        self.Payload = payload
        self.IssuedAt = time.time()
        self.NotBefore = not_before if (not_before is None or not_before > 365*24*3600) else self.IssuedAt + not_before
        self.Expiration = expiration if (expiration is None or expiration > 365*24*3600) else self.IssuedAt + expiration
        
    def __str__(self):
        return "SignedToken(alg=%s, iat=%s, nbf=%s, exp=%s, payload=%s)" % (self.Alg, self.IssuedAt, self.NotBefore, self.Expiration,
            self.Payload)
        
    @staticmethod
    def encode_object(x):
        j = json.dumps(x)
        if isinstance(j, str):
            j = j.encode("utf-8")
        return base64.b64encode(j)
        
    @staticmethod
    def decode_object(txt):
        return json.loads(base64.b64decode(txt))
        
    @staticmethod
    def pack(*words):
        assert len(words) == 3, "Token must consist of 3 words, got %d instead" % (len(words),)
        words = [w.encode("utf-8") if isinstance(w, str) else w for w in words]
        return b".".join(words)
        
    @staticmethod
    def unpack(txt):
        words = txt.split(b'.')
        assert len(words) == 3, "Token must consist of 3 words, got %d instead: [%s]" % (len(words), txt)
        return words
    
    @staticmethod
    def signature(alg, *words):
        text = SignedToken.pack(*words)
        h = hashlib.new(alg)
        h.update(text)
        sig = h.hexdigest()
        if isinstance(sig, str):
                sig = sig.encode("utf-8")
        return sig

    def encode(self, secret):
        header = self.encode_object(
            {"iat":self.IssuedAt, "exp":self.Expiration, "alg":self.Alg, "nbf":self.NotBefore}
        )
        payload = self.encode_object(self.Payload)
        signature = self.signature(self.Alg, header, payload, secret)
        return self.pack(header, payload, signature)
        
    @staticmethod
    def decode(txt, secret=None, verify_times=False, leeway=0):
        header, payload, signature = SignedToken.unpack(txt)
        #print ("token.decode:", header, payload, signature)
        header_decoded = SignedToken.decode_object(header)
        try:    alg = header_decoded["alg"]
        except: raise SignedTokenHeaderError
        exp = header_decoded.get("exp")
        nbf = header_decoded.get("nbf")
        iat = header_decoded.get("iat")
        if secret is not None:
            if not alg in SignedToken.AcceptedAlgorithms:
                raise SignedTokenUnacceptedAlgorithmError(alg)
            calculated_signature = SignedToken.signature(alg, header, payload, secret)
            if calculated_signature != signature:
                raise SignedTokenSignatureVerificationError
        if verify_times:
            if exp is not None and time.time() > exp + leeway:
                raise SignedTokenExpiredError
            if nbf is not None and time.time() < nbf - leeway:
                raise SignedTokenImmatureError
        payload = SignedToken.decode_object(payload)
        
        token = SignedToken(payload, exp, nbf)
        token.IssuedAt = iat
        token.Alg = alg
        return token
        
class TokenBox(object):
    def __init__(self, url, username, password, margin = 10, request_now = False):
        self.URL = url
        self.Username = username
        self.Password = password
        self.Token = None
        self.Expiration = 0
        self.Encoded = None
        self.Margin = margin
        self.Identity = None
        if request_now:
            self.renewIfNeeded()
        
    def renewIfNeeded(self):
        need_to_renew = self.Token is None or time.time() > self.Expiration - self.Margin
        if need_to_renew:
            status, body = digest_client(self.URL, self.Username, self.Password)
            if status/100 == 2:
                encoded = body.strip()
                t = SignedToken.decode(encoded)
                self.Token = t
                self.Encoded = encoded
                self.Expiration = t.Expiration
                self.Identity = t.Payload.get("identity", "")
            else:
                raise SignedTokenAuthoriztionError(body)
    
    @property
    def token(self):
        self.renewIfNeeded()
        return self.Encoded
            
        
if __name__ == "__main__":
    import os
    secret = generate_secret(128)
    payload = {"text":"hello world", "temp":32.0}
    encoded = SignedToken(payload, expiration=0).encode(secret)
    
    print(encoded)
    
    t1 = SignedToken.decode(encoded, secret, leeway=10)
    print(t1)
    print(t1.Payload)
        
    
        
