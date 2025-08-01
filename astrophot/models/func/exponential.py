import torch
from .sersic import sersic_n_to_b

b = sersic_n_to_b(1.0)


def exponential(R: torch.Tensor, Re: torch.Tensor, Ie: torch.Tensor) -> torch.Tensor:
    """Exponential 1d profile function, specifically designed for pytorch
    operations.

    **Args:**
    -  `R`: Radius tensor at which to evaluate the exponential function
    -  `Re`: Effective radius in the same units as R
    -  `Ie`: Effective surface density
    """
    return Ie * torch.exp(-b * ((R / Re) - 1.0))
