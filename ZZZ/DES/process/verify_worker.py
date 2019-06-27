import numpy as np, math

class Worker(object):

    Columns = ["ALPHAWIN_J2000", "DELTAWIN_J2000", "HPIX", "OBJECT_ID",
        "Observation.HPIX", "Observation.ALPHAWIN_J2000",
        "Observation.DELTAWIN_J2000", "Observation.OBJECT_ID"]

    Delta = 1.0/3600.0        # 1 arcsecond

    def dist(self, ra1, dec1, ra2, dec2):
        return max(abs(ra1-ra2), abs(dec1-dec2))

    def dist_arr(self, ra1, dec1, ra2, dec2):
        return max(np.max(np.abs(ra1-ra2)), np.max(np.abs(dec1-dec2)))
    
    def close(self, ra1, dec1, ra2, dec2):
        return self.dist(ra1, dec1, ra2, dec2) < self.Delta

    def run(self, catalog, job):
    
        rgid = catalog.rgid
        
        #
        # Objects stored by HPIX ?
        #
        
        if np.any(catalog.HPIX[1:] < catalog.HPIX[:-1]):
            job.send(message="Objects in RG %d not sorted by HPIX" % (rgid,))
        
        for obj in catalog:
            obj_ra, obj_dec = obj.ALPHAWIN_J2000, obj.DELTAWIN_J2000
            obs = obj.Observation
            
            #
            # object to observations distance
            #
            dist = self.dist_arr(obj_ra, obj_dec, 
                obs.ALPHAWIN_J2000,
                obs.DELTAWIN_J2000)
            if dist > self.Delta:
                job.send(message="Angular distance between object %d in RG %d and one of observations > %f\n  object RA:%f Dec:%f\n  observations RA:%s\n  observations Dec:%s" %
                    (obj.OBJECT_ID, rgid, self.Deltai, obj_ra, obj_dec, obs.ALPHAWIN_J2000, obs.DELTAWIN_J2000))
        
            #
            # Check if all observations have the same HPIX and the object
            #
            if not np.all(obs.HPIX == obj.HPIX):
                job.send(message="Object %d HPIX mismatch with one of its observations in RG %d" %
                    (obj.OBJECT_ID, rgid))
                   
            #
            # Object id == observation.OBJECT_ID ?
            #          
            if not np.all(obs.OBJECT_ID == obj.OBJECT_ID):
                job.send(message="Object %d ID mismatch with one of its observations in RG %d" %
                    (obj.OBJECT_ID,rgid))
    
    
 
