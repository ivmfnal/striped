import numpy as np, math

class Worker(object):

    Columns = ["ALPHAWIN_J2000", "DELTAWIN_J2000", "COADD_OBJECT_ID"]
    Delta = 1.0/3600.0        # 1 arcsecond

    def dist(self, ra1, dec1, ra2, dec2):
        return max(abs(ra1-ra2), abs(dec1-dec2))
    
    def close(self, ra1, dec1, ra2, dec2):
        return self.dist(ra1, dec1, ra2, dec2) < self.Delta

    def run(self, catalog, job):
        observations = job.InputData
        # assume oboth observations and objects are sorted by hpix index

        c_ra = catalog.ALPHAWIN_J2000
        c_dec = catalog.DELTAWIN_J2000
        c_id = catalog.COADD_OBJECT_ID
        ncatalog = len(c_ra)

	o_ra = observations[:,0]
	o_dec = observations[:,1]

	c_ra_min, c_ra_max = min(c_ra), max(c_ra)
	c_dec_min, c_dec_max = min(c_dec), max(c_dec)

	o_ra_min, o_ra_max = min(o_ra), max(o_ra)
	o_dec_min, o_dec_max = min(o_dec), max(o_dec)

	if c_ra_min > o_ra_max or c_ra_max < o_ra_min or \
			c_dec_min > o_dec_max or c_dec_max < o_dec_min:
		job.send(overlap = np.array([0]))
		job.send(ranges = np.array([
			(
				(c_ra_min, c_ra_max),
				(c_dec_min, c_dec_max)
			),
			(
				(o_ra_min, o_ra_max),
				(o_dec_min, o_dec_max)
			)
		]))
	
		return
		
	job.send(overlap = np.array([1]))
	
        
        matches = []         # (inx, object_id)
        match_angles = []
        unmatched = []       # inx
        #print hp0, hp1, c_hpix[0], c_hpix[-1]
	
	for io, observation in enumerate(observations):
		for ic, c in enumerate(catalog):
			if self.close(c.ALPHAWIN_J2000, c.DELTAWIN_J2000, observation[0], observation[1]):
				matches.append((io, ic))
				break
		else:
			unmatched.append(io)
        #
        # phase2 here
        #
        if matches:
            job.send(matches = np.array(matches), match_angles = np.array(match_angles))
