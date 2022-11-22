import torch
import numpy as np
from autoprof.utils.conversions.functions import sersic_n_to_b

def sersic_torch(R, n, Re, Ie):
    """Seric 1d profile function, specifically designed for pytorch
    operations

    """
    bn = sersic_n_to_b(n)
    return Ie * torch.exp(-bn*(torch.pow((R+1e-6)/Re, 1/n) - 1)) # epsilon added for numerical stability of gradient
def sersic_np(R, n, Re, Ie):
    """Sersic 1d profile function, works more generally with numpy
    operations. In the event that impossible values are passed to the
    function it returns large values to guide optimizers away from
    such values.

    """
    if np.any(np.array([n, Re, Ie]) <= 0):
        return np.ones(len(R))*1e6
    bn = sersic_n_to_b(n)
    return Ie*np.exp(-bn*((R/Re)**(1/n) - 1))

def gaussian_torch(R, sigma, I0):
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    """
    return (I0 / torch.sqrt(2 * np.pi * sigma**2)) * torch.exp(-0.5*torch.pow(R/sigma,2))
def gaussian_np(R, sigma, I0):
    """Gaussian 1d profile function, works more generally with numpy
    operations.

    """
    return (I0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5*((R/sigma)**2))

def exponential_torch(R, Re, Ie):
    """Exponential 1d profile function, specifically designed for pytorch
    operations.

    """
    return Ie * torch.exp(- sersic_n_to_b(1.) * ((R / Re) - 1.))
def exponential_np(R, Ie, Re):
    """Exponential 1d profile function, works more generally with numpy
    operations.

    """
    return Ie * np.exp(- sersic_n_to_b(1.) * (R / Re - 1.))
