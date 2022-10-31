import numpy as np
import torch
from scipy.special import gamma
from torch.special import gammaln

def sersic_I0_to_flux_np(I0, n, R, q):
    return 2*np.pi*I0*q*n*R**2 * gamma(2*n)
def sersic_flux_to_I0_np(flux, n, R, q):
    return flux / (2*np.pi*q*n*R**2 * gamma(2*n))

def sersic_I0_to_flux_torch(I0, n, R, q):
    return 2*np.pi*I0*q*n*R**2 * torch.exp(gammaln(2*n))
def sersic_flux_to_I0_torch(flux, n, R, q):
    return flux / (2*np.pi*q*n*R**2 * torch.exp(gammaln(2*n)))
