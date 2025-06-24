import torch
import numpy as np

sq_2pi = np.sqrt(2 * np.pi)


def gaussian(R, sigma, flux):
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        sigma: standard deviation of the gaussian in the same units as R
        I0: central surface density
    """
    return (flux / (sq_2pi * sigma)) * torch.exp(-0.5 * torch.pow(R / sigma, 2))
