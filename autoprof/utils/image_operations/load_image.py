import numpy as np
from astropy.io import fits

def load_fits(filename, hduelement = 0):
    hdul = fits.open(filename)
    dat = hdul[hduelement].data
    return np.require(dat, dtype=float)
    
def load_npy(filename):
    dat = np.load(filename)
    return np.require(dat, dtype=float)

def load_png(filename):
    pass
