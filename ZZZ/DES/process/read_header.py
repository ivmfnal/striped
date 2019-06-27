import glob
import fitsio
import numpy as np
import numpy.lib.recfunctions as recfuncs
from astropy.io.fits import Header

# Read the catalog FITS file
filename = '/data/bliss/566500/566509/D00566509_i_10_r1p1_fullcat.fits'
fits = fitsio.FITS(filename)

# The header for the image that this catalog was derived from is
# stored in the first extension of the FITS file. However, it's format
# is weird, so we parse it.

# First we build a string
hdrstr = '\n'.join(fits['LDAC_IMHEAD'].read()[0][0])
# Then we use astropy to parse that string into a dict
hdr = Header.fromstring(hdrstr,sep='\n')

# Now we read the catalog
catalog = fits['LDAC_OBJECTS'].read()

# The image header gives us access to image-level quantities, like
# EXPNUM, CCDNUM, MJD-OBS, etc. Careful, these quantities may have a different byte order than the catalog data. 
EXPNUM = np.tile(hdr['EXPNUM'],len(catalog))
CCDNUM = np.tile(hdr['CCDNUM'],len(catalog))

# We can then append those quantities to the object array
data = recfuncs.rec_append_fields(catalog,
                                  names=['EXPNUM','CCDNUM'],
                                  data=[EXPNUM,CCDNUM])

print("Check out the byte order...")
print(data.dtype.descr[-4:])
print data['EXPNUM'],data['CCDNUM']
