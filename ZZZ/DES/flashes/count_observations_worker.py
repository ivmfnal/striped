import numpy as np, math

class Worker(object):

    Columns = ["HPIX","Observation.HPIX"]

    def run(self, objects, job, db):
        min_objects = job["min_objects"]
        objects_hpix, objects_hpix_counts = np.unique(objects.HPIX, return_counts=True)   
        obs_hpix, obs_hpix_counts = np.unique(objects.Observation.HPIX, return_counts=True)
        assert np.all(objects_hpix==obs_hpix)        # naive, assumes object.hpix == observation.hpix always

        good_pixels = objects_hpix_counts >= min_objects
        
        objects_hpix, objects_hpix_counts, obs_hpix_counts = objects_hpix[good_pixels], objects_hpix_counts[good_pixels], obs_hpix_counts[good_pixels]
        
        job.send(
            hpix_stats = (objects_hpix, objects_hpix_counts, obs_hpix_counts)
        )
            
