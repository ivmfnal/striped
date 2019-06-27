import fitsio, sys

data = fitsio.read(sys.argv[1], columns=["ALPHAWIN_J2000", "DELTAWIN_J2000"])
ra_min, ra_max = min(data["ALPHAWIN_J2000"]), max(data["ALPHAWIN_J2000"])
dec_min, dec_max = min(data["DELTAWIN_J2000"]), max(data["DELTAWIN_J2000"])

print ra_min, ra_max, dec_min, dec_max

