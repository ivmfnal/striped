import numpy as np, math

class Worker(object):

    Columns = ["HPIX","Observation.HPIX"]

    def run(self, objects, job):
        objects_hpix, objects_hpix_counts = np.unique(objects.HPIX, return_counts=True)   
        obs_hpix, obs_hpix_counts = np.unique(objects.Observation.HPIX, return_counts=True)
        asert np.all(objects_hpix==obs_hpix)        # naive
        counts = np.array(
        job.send(
            hpix_mean_counts = (objects_hpix, objects_hpix_counts, obs_hpix_counts)
        )
            
