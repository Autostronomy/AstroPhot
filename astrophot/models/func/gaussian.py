import torch
import numpy as np


def gaussian(R, sigma, I0):
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        sigma: standard deviation of the gaussian in the same units as R
        I0: central surface density
    """
    return (I0 / torch.sqrt(2 * np.pi * sigma**2)) * torch.exp(-0.5 * torch.pow(R / sigma, 2))
