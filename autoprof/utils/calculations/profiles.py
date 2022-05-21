import numpy as np

def sersic(R, n, Re, Ie):
    bn = 2*n-1/3
    return Ie*np.exp(-bn*((R/Re)**(1/n) - 1))

def gaussian(R, I0, sigma):
    return I0 * np.exp(-0.5*((R/sigma)**2))
