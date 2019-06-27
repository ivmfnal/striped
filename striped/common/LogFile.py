import time
import os
import string
import datetime
from threading import RLock

def synchronized(func):
	def synced(self, *params, **args):
		with self._Lock:
			return func(self, *params, **args)
	return synced

class   LogFile:
        def __init__(self, path, interval = '1d', keep = 10, timestamp=True, append=True):
                # interval = 'midnight' means roll over at midnight
		self._Lock = RLock()
                self.Path = path
                self.File = None
                self.CurLogBegin = 0
                if type(interval) == type(''):
                        mult = 1
                        if interval[-1] == 'd' or interval[-1] == 'D':
                                interval = interval[:-1]
                                mult = 24 * 3600
                                interval = string.atoi(interval) * mult
                        elif interval[-1] == 'h' or interval[-1] == 'H':
                                interval = interval[:-1]
                                mult = 3600
                                interval = string.atoi(interval) * mult
                        elif interval[-1] == 'm' or interval[-1] == 'M':
                                interval = interval[:-1]
                                mult = 60
                                interval = string.atoi(interval) * mult
                self.Interval = interval
                self.Keep = keep
                self.Timestamp = timestamp
                self.LineBuf = ''
                self.LastLog = None
                if append:
                    self.File = open(self.Path, 'a')
                    self.CurLogBegin = time.time()
                        
        def newLog(self):
                if self.File != None:
                        self.File.close()
                try:    os.remove('%s.%d' % (self.Path, self.Keep))
                except: pass
                for i in range(self.Keep - 1):
                        inx = self.Keep - i
                        old = '%s.%d' % (self.Path, inx - 1)
                        new = '%s.%d' % (self.Path, inx)
                        try:    os.rename(old, new)
                        except: pass
                try:    os.rename(self.Path, self.Path + '.1')
                except: pass
                self.File = open(self.Path, 'w')
                self.CurLogBegin = time.time()
                
        @synchronized
        def log(self, msg):
                t = time.time()
                if self.Timestamp:
                        tm = time.localtime(t)
                        msg = time.strftime('%D %T: ', tm) + msg
                if self.Interval == 'midnight':
                        if datetime.date.today() != self.LastLog:
                                self.newLog()
                elif isinstance(self.Interval,int):
                        if t > self.CurLogBegin + self.Interval:
                                self.newLog()
                self.File.write(msg + '\n');
                self.File.flush()
                self.LastLog = datetime.date.today()

        @synchronized
        def     write(self, msg):
                self.LineBuf = self.LineBuf + msg
                inx = string.find(self.LineBuf, '\n')
                while inx >= 0:
                        self.log(self.LineBuf[:inx])
                        self.LineBuf = self.LineBuf[inx+1:]
                        inx = string.find(self.LineBuf, '\n')

        @synchronized
        def     flush(self):
                if self.LineBuf:
                        self.log(self.LineBuf)
                        self.LineBuf = ''
                
