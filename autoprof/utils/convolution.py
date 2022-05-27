import numpy as np
from astropy.convolution import convolve, convolve_fft

def direct_convolve(img, psf, mask = None):
    
    return convolve(img, psf, boundary = 'extend', mask = mask)

def fft_convolve(img, psf, mask = None):

    return convolve_fft(img, psf, mask = mask)
