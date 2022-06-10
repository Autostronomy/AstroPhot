import numpy as np

def sersic(R, n, Rs, I0):
    return I0*np.exp(-(R/Rs)**(1/n))

def gaussian(R, I0, sigma):
    return I0 * np.exp(-0.5*((R/sigma)**2))

def exponential(R, I0, Rd):
    return I0 * np.exp(R / Rd)
