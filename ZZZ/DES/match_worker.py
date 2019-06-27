import numpy as np, math

class Worker(object):

    Columns = ["RA", "Dec", "HPix", "COADD_object_id"]
    Delta = 1.0/3600.0        # 1 arcsecond

    def dist(self, ra1, dec1, ra2, dec2):
        return max(abs(ra1-ra2), abs(dec1-dec2))
    
    def close(self, ra1, dec1, ra2, dec2):
        return self.dist(ra1, dec1, ra2, dec2) < self.Delta

    def run(self, catalog, job):
        observations = job.InputData
        hpix = np.asarray(observations[:,2], np.int)            # [:,(ra, dec, hpix)]
        # assume oboth observations and objects are sorted by hpix index
        hp0, hp1 = hpix[0], hpix[-1]
        c_hpix = catalog.HPix
        c_ra = catalog.RA
        c_dec = catalog.Dec
        c_id = catalog.COADD_object_id
        ncatalog = len(c_hpix)
        
        matches = []         # (inx, object_id)
        match_angles = []
        unmatched = []       # inx
        #print hp0, hp1, c_hpix[0], c_hpix[-1]
        if hp0 <= c_hpix[-1] and hp1 >= c_hpix[0]:
            # find first possible hpix match in the catalog
            i_c = 0
            for i_o, o in enumerate(observations):
                o_hpix = hpix[i_o]
                o_ra = o[0]
                o_dec = o[1]
                id_ed = False
                while i_c < ncatalog and c_hpix[i_c] < o_hpix:
                    i_c += 1
                found = False
                if i_c < ncatalog and c_hpix[i_c] == o_hpix:
                    j = i_c
                    best_dist = None
                    oid = None
                    best_ra, best_dec = None, None
                    while not found and j < ncatalog and c_hpix[j] == o_hpix:
                        if self.close(o_ra, o_dec, c_ra[j], c_dec[j]):
                            dist = self.dist(o_ra, o_dec, c_ra[j], c_dec[j])
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                oid = c_id[j]
                                best_ra, best_dec = c_ra[j], c_dec[j]
                        j += 1
                    if oid != None:
                        matches.append((i_o, oid))
                        match_angles.append([o_ra, o_dec, best_ra, best_dec])
                        found = True
                if not found:
                    unmatched.append(i_o)
        #
        # phase2 here
        #
        if matches:
            job.send(matches = np.array(matches), match_angles = np.array(match_angles))
