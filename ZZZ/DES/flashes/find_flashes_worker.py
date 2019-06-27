import numpy as np, math

class Worker(object):

    Columns = ["HPIX", "OBJECT_ID", 
        "Observation.ALPHAWIN_J2000", "Observation.OBJECT_ID", "Observation.HPIX",
        "Observation.DELTAWIN_J2000", "Observation.CCDNUM", "Observation.EXPNUM"]

    def run(self, objects, job, db):
        pixels = job["pixels"]

        min_hpix = pixels[0]
        max_hpix = pixels[-1]

        filter1 = objects.filter(np.isin(objects.HPIX, pixels))
        r1 = filter1.ratio
        
        objects = filter1(objects)
        
        filter2 = objects.filter(objects.Observation.count == 1)
        r2 = filter2.ratio
        
        objects = filter2(objects)
        
        if len(objects) > 0:
            obs = objects.Observation

            flashes = np.array(
                zip(obs.OBJECT_ID, obs.HPIX, obs.ALPHAWIN_J2000, obs.DELTAWIN_J2000, obs.CCDNUM, obs.EXPNUM),
                dtype=[("OBJECT_ID",np.int64), ("HPIX",np.int64), ("ALPHAWIN_J2000", np.float64), ("DELTAWIN_J2000", np.float64),
                    ("CCDNUM",np.int64),("EXPNUM",np.int64)])

            #job.message("worker: found %d flashes in %d" % (len(flashes), objects.rgid))
            job.send(flashes = flashes, rgid=objects.rgid)

