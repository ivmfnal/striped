import time, sys

class Stopwatch(object):

    def __init__(self, label = None, inline=False):
        self.T0 = None
        self.T = 0
        self.Label = label
        self.Inline = inline

    def __enter__(self):
        self.T0 = time.time()
        if self.Inline:
            print(self.Label, "...", end=' ')
            sys.stdout.flush()
        return self

    def __exit__(self, *args):
        self.T = time.time() - self.T0
        if self.Label:
            if not self.Inline:
                print(self.Label, end=' ')
            print(("%.6f" % (self.T,)))


    def __str__(self):
        return str(self.T)
