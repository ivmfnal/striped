import numpy as np, math

class Worker(object):

    Columns = ["HPIX"]

    def run(self, objects, job):
        hpix = objects.HPIX
        job.send(hpix=np.array(sorted(list(set(hpix)))))
    
