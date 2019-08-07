from stripe_tools import pairs
import numpy as np
import random, time

nevents = 10000
nitems = 10
tsum = 0.0
nframes = 100

for iframe in range(nframes):
    # generate events
    stripe = []
    size = []
    for ievent in range(nevents):
        n = random.randint(0,nitems)
        stripe += [random.random() for _ in range(n)]
        size.append(n)
    stripe = np.array(stripe)
    size = np.array(size)
    
    t0 = time.time()
    pairs_data, pairs_sizes = pairs(stripe, size)
    tsum += time.time() - t0    
    
print((tsum/nframes))
