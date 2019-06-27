from trace import Tracer
from Q import Dataset
from Q import cosh, sqrt, cos
from StripedClient import StripedClient
import sys, math, traceback, time, os
import numpy as np

columns = ["Muon.pt", "Muon.eta", "Muon.phi", "Muon.q"]
T = Tracer(calibrate=True)

url = sys.argv[1]
dataset_name = sys.argv[2]

client = StripedClient(sys.argv[1])

dataset = Dataset(client, dataset_name, columns, trace = T)

#
# init
#

m = dataset.event.Muon
m.p = m.pt * cosh(m.eta)

pair = dataset.event.Muon.pair
m1, m2 = pair
pair.M = sqrt(2*m1.pt*m2.pt*(cosh(m1.eta-m2.eta) - cos(m1.phi-m2.phi)))  
pair.C = m1.q * m2.q

with T["run"]:
    nevents = 0
    for e in dataset.events(range(100)):
        with T["run/event"]:
            for pair in e.Muon.pairs.iterate():
                with T["run/event/pair"]:
                    with T["run/event/pair/unpack"]:
                        m1, m2 = pair.asTuple()
                    with T["run/event/pair/analyze"]:
                        if pair.C < 0 and m1.p < 1000 and m2.p < 1000:     
                            with T["run/event/pair/analyze/getM"]:
                                M = pair.M  
                            if M < 120 and M > 60:
                                if M < 92 and M > 88:
                                    pass
        nevents += 1
        if nevents >= 100000:
            break                    
                
T.printStats()
