from StripedClient import StripedClient
from Coordinator import Coordinator
import sys, os, time, getopt
import numpy as np
from MyThread import Lockable, synchronized
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from cStringIO import StringIO

from striped_job import StripedJob

class DimuonJob(StripedJob):

    def __init__(self):
        StripedJob.__init__(self)

        self.Masses = []
    
    def dataReceived(self, wid, data):
        if len(data) != 1000:
            print "received %d entries from %s" % (len(data), wid)
        self.Masses += list(data)

    def initDisplay(self, figure):
        self.Fig = figure
        self.MassHist = self.Fig.add_subplot(1,1,1,)
        self.Bins = 120
        
    def updateDisplay(self, iframe):
        self.MassHist.clear()
        self.MassHist.hist(self.Masses, bins=self.Bins)
        self.MassHist.set_title(r"Mass($\mu^+,\mu^-$)")

    def jobFinished(self):
        print "Receined %d entries" % (len(self.Masses),)
        print "-- done --"

if __name__ == "__main__":

    nworkers = 1

    opts, args = getopt.getopt(sys.argv[1:], "n:")
    for opt, val in opts:
            if opt == "-n": nworkers = int(val)

    URLHead = args[0]
    DatasetName = args[1]
    WorkerModule = args[2]

    job = DimuonJob()
    job.start(URLHead, DatasetName, WorkerModule, nworkers=nworkers)
