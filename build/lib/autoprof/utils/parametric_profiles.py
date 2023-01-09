import torch
import numpy as np
from .conversions.functions import sersic_n_to_b
from .interpolate import cubic_spline_torch, interp1d_torch

def sersic_torch(R, n, Re, Ie):
    """Seric 1d profile function, specifically designed for pytorch
    operations

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        n: sersic index restricted to n > 0.36
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    bn = sersic_n_to_b(n)
    return Ie * torch.exp(-bn*(torch.pow((R+1e-8)/Re, 1/n) - 1)) # epsilon added for numerical stability of gradient
def sersic_np(R, n, Re, Ie):
    """Sersic 1d profile function, works more generally with numpy
    operations. In the event that impossible values are passed to the
    function it returns large values to guide optimizers away from
    such values.

    Parameters:
        R: Radii array at which to evaluate the sersic function
        n: sersic index restricted to n > 0.36
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    if np.any(np.array([n, Re, Ie]) <= 0):
        return np.ones(len(R))*1e6
    bn = sersic_n_to_b(n)
    return Ie*np.exp(-bn*((R/Re)**(1/n) - 1))

def gaussian_torch(R, sigma, I0):
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        sigma: standard deviation of the gaussian in the same units as R
        I0: central surface density
    """
    return (I0 / torch.sqrt(2 * np.pi * sigma**2)) * torch.exp(-0.5*torch.pow(R/sigma,2))
def gaussian_np(R, sigma, I0):
    """Gaussian 1d profile function, works more generally with numpy
    operations.

    Parameters:
        R: Radii array at which to evaluate the sersic function
        sigma: standard deviation of the gaussian in the same units as R
        I0: central surface density
    """
    return (I0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5*((R/sigma)**2))

def exponential_torch(R, Re, Ie):
    """Exponential 1d profile function, specifically designed for pytorch
    operations.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    return Ie * torch.exp(- sersic_n_to_b(torch.tensor(1.,dtype = R.dtype, device = R.device)) * ((R / Re) - 1.))

def exponential_np(R, Ie, Re):
    """Exponential 1d profile function, works more generally with numpy
    operations.

    Parameters:
        R: Radii array at which to evaluate the sersic function
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    return Ie * np.exp(- sersic_n_to_b(1.) * (R / Re - 1.))

def nonparametric_torch(R, profR, profI, pixelscale2, extend):
    """Nonparametric 1d profile function, cubic spline between points up
    to second last point beyond which is linear, specifically designed
    for pytorch.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        profR: radius values for the surface density profile in the same units as R
        profI: surface density values for the surface density profile
        pixelscale2: squared pixelscale. Just use 1 if not needed
    """
    I = cubic_spline_torch(profR, profI, R.view(-1), extend = "none").view(*R.shape)
    res = torch.zeros_like(I)
    res[R <= profR[-2]] = 10**(I[R <= profR[-2]])
    if extend:
        res[R > profR[-2]] = 10**(profI[-2] + (R[R > profR[-2]] - profR[-2])*((profI[-1] - profI[-2])/(profR[-1] - profR[-2])))
    else:
        res[R > profR[-2]] = 0
    return res * pixelscale2
    
