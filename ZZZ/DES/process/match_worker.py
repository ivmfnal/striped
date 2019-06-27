import numpy as np, math

class Worker(object):

    Columns = ["ALPHAWIN_J2000", "DELTAWIN_J2000", "HPIX", "OBJECT_ID"]
    Delta = 1.0/3600.0        # 1 arcsecond

    def dist(self, ra1, dec1, ra2, dec2):
        return max(abs(ra1-ra2), abs(dec1-dec2))
    
    def close(self, ra1, dec1, ra2, dec2):
        return self.dist(ra1, dec1, ra2, dec2) < self.Delta

    def run(self, catalog, job):
        observations = job["observations"]
        o_hpix = observations["HPIX"]           
        #print o_hpix[:100]
        # assume oboth observations and objects are sorted by hpix index
        o_hp0, o_hp1 = o_hpix[0], o_hpix[-1]
        c_hpix = catalog.HPIX
        c_hp0, c_hp1 = c_hpix[0], c_hpix[-1]

        #print "input hpix range:  ", hpix[0], hpix[-1]
        #print "catalog hpix range:", c_hpix[0], c_hpix[-1]

        c_ra = catalog.ALPHAWIN_J2000
        c_dec = catalog.DELTAWIN_J2000
        c_id = catalog.OBJECT_ID
        ncatalog = len(c_hpix)
        
        matches = []         # (inx, object_id)
        unmatched = []       # inx
        #print hp0, hp1, c_hpix[0], c_hpix[-1]
        
        #print "c_hpix:", c_hpix[:100]
        #print "o_hpix:", o_hpix[:100]
        
        if o_hp0 <= c_hp1 and o_hp1 >= c_hp0:
            # find first possible hpix match in the catalog

            this_hpix_inx = 0
        
            for i_o, (o_ra, o_dec, o_hpix) in enumerate(observations):

                # find hpix in the catalog
                while this_hpix_inx < ncatalog and c_hpix[this_hpix_inx] < o_hpix:
                    this_hpix_inx += 1
                if this_hpix_inx >= ncatalog:
                    break       # observation hpix > last catalog hpix
                this_hpix = c_hpix[this_hpix_inx]

                i = this_hpix_inx
                best_dist = None
                oid = None
                best_ra, best_dec = None, None
                while i < ncatalog and c_hpix[i] == o_hpix:
                    if self.close(o_ra, o_dec, c_ra[i], c_dec[i]):
                        dist = self.dist(o_ra, o_dec, c_ra[i], c_dec[i])
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            oid = c_id[i]
                            best_ra, best_dec = c_ra[i], c_dec[i]
                    i += 1
                if oid != None:
                    #job.send(message="Match found %d:%d in RG %d" % (i_o, oid, catalog.rgid))
                    matches.append((i_o, oid, catalog.rgid, best_dist))
        #
        # phase2 here
        #
        if matches:
            array = np.array(matches, dtype=[("index",np.int64),("object_id",np.int64),("rgid",np.int64),("best_dist",np.float64)])           # [n,3]: (inx, oid, rgid)
            # filter matches in such a way that only one observation can match an object. Choose the observation by the distance.
            array = np.sort(array, order=["object_id","best_dist"])
            n = len(array)
            keep_inx = np.ones((n,), dtype=np.bool)
            keep_inx[1:] = array["object_id"][1:] != array["object_id"][:-1]
            array = array[keep_inx]
            job.send(matches = array)
