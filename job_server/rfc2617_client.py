from urllib2 import Request, urlopen, HTTPPasswordMgr, HTTPDigestAuthHandler, build_opener
import sys



host, port, user, password = sys.argv[1:]
port = int(port)
url = "http://%s:%s/hello" % (host, port)

"""
pwdmgr = HTTPPasswordMgr()
pwdmgr.add_password("secret", "/hello", user, password)
handler = HTTPDigestAuthHandler(pwdmgr)
opener = build_opener(handler)

resp = opener.open(url)

print resp
"""


import requests
from requests.auth import HTTPDigestAuth

resp = requests.get(url, auth=HTTPDigestAuth(user, password))
print resp


