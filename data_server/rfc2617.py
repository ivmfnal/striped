import sys
import re
from pprint import pprint
from base64 import *
from time import time
import md5
from wsgiref.simple_server import make_server
import wsgiref.headers

def md5sum(data):
  m = md5.new()
  m.update(data)
  return m.hexdigest()

def calc_a1(user='test'):
  return md5sum('%s:%s:%s' % (user, 'secret', 'test'))          # user:realm:password

def check_authorization(env, resp):
  matches = re.compile('Digest \s+ (.*)', re.I + re.X).match(resp)
  if not matches:
    return None

  vals = re.compile(', \s*', re.I + re.X).split(matches.group(1))

  dict = {}

  pat = re.compile('(\S+?) \s* = \s* ("?) (.*) \\2', re.X)
  for val in vals:
    ms = pat.match(val)
    if not ms:
      raise 'ERROR'
    dict[ms.group(1)] = ms.group(3)

  #assert algorithm=='MD5', qop=='auth', ...
  #assert username=='test'?

  pprint(dict, sys.stderr, 2)

  a1 = calc_a1(dict['username'])
  a2 = md5sum('%s:%s' % (env['REQUEST_METHOD'], dict['uri']))
  myresp = md5sum('%s:%s:%s:%s:%s:%s' % (a1, dict['nonce'], dict['nc'], dict['cnonce'], dict['qop'], a2))
  pprint(myresp, sys.stderr, 2)
  if myresp != dict['response']:
    print >>sys.stderr, "Auth failed!"
    return None

  # TODO: check nonce's timestamp
  cur_nonce = int(time())
  aut_nonce = int(b64decode(dict['nonce']))
  pprint({'cli': aut_nonce, 'srv': cur_nonce}, sys.stderr, 2)
  if cur_nonce - aut_nonce > 10:    # 10sec
    print >>sys.stderr, "Too old!"
    return False

  return dict['username']

def app(environ, start_response):
  heads = wsgiref.headers.Headers([])

  heads.add_header('Content-Type', 'text/plain')

  auth = environ.get('HTTP_AUTHORIZATION', '')
  state = check_authorization(environ, auth)
  if state:
    start_response('200 OK', heads.items())
    return ['OK!']

  nonce = b64encode(str(int(time())))
  auth_head = 'Digest realm="secret", nonce="%s", algorithm=MD5, qop="auth"' % (nonce)
  if state == False:
    auth_head += ', stale=true'
  heads.add_header('WWW-Authenticate', auth_head)
  start_response('401 Authorization Required', heads.items())

  return ['Hello, World!']
