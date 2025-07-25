import torch
from .sersic import sersic_n_to_b

b = sersic_n_to_b(1.0)


def exponential(R, Re, Ie):
    """Exponential 1d profile function, specifically designed for pytorch
    operations.

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    return Ie * torch.exp(-b * ((R / Re) - 1.0))
