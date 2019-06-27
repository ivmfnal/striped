import glob
import fitsio
import numpy as np

# Find the catalogs for all CCDs in the exposure
filenames = sorted(glob.glob('/data/kadrlica/bliss/566500/566500/*_fullcat.fits'))
# Read only the position columns
columns = ['ALPHAWIN_J2000','DELTAWIN_J2000']
# Read each CCD and concatenate into one array
data = np.concatenate([fitsio.read(f,ext=2,columns=columns) for f in filenames])

