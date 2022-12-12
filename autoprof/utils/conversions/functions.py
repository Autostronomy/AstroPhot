import numpy as np
import torch
from scipy.special import gamma
from torch.special import gammaln

def sersic_n_to_b(n):
    return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)

def sersic_I0_to_flux_np(I0, n, R, q):
    return 2*np.pi*I0*q*n*R**2 * gamma(2*n)
def sersic_flux_to_I0_np(flux, n, R, q):
    return flux / (2*np.pi*q*n*R**2 * gamma(2*n))
def sersic_Ie_to_flux_np(Ie, n, R, q):
    bn = sersic_n_to_b(n)
    return 2*np.pi*Ie*R**2*q*n * (np.exp(bn)*bn**(-2*n)) * gamma(2*n)
def sersic_flux_to_Ie_np(flux, n, R, q):
    bn = sersic_n_to_b(n)
    return flux / (2*np.pi*R**2*q*n * (np.exp(bn)*bn**(-2*n)) * gamma(2*n))

def sersic_I0_to_flux_torch(I0, n, R, q):
    return 2*np.pi*I0*q*n*R**2 * torch.exp(gammaln(2*n))
def sersic_flux_to_I0_torch(flux, n, R, q):
    return flux / (2*np.pi*q*n*R**2 * torch.exp(gammaln(2*n)))
def sersic_Ie_to_flux_torch(Ie, n, R, q):
    bn = sersic_n_to_b(n)
    return 2*np.pi*Ie*R**2*q*n * (torch.exp(bn)*bn**(-2*n)) * torch.exp(gammaln(2*n))
def sersic_flux_to_Ie_torch(flux, n, R, q):
    bn = sersic_n_to_b(n)
    return flux / (2*np.pi*R**2*q*n * (torch.exp(bn)*bn**(-2*n)) * torch.exp(gammaln(2*n)))
