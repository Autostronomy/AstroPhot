import torch
import numpy as np

sq_2pi = np.sqrt(2 * np.pi)


def gaussian(R: torch.Tensor, sigma: torch.Tensor, flux: torch.Tensor) -> torch.Tensor:
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    **Args:**
    -  `R`: Radii tensor at which to evaluate the gaussian function
    -  `sigma`: Standard deviation of the gaussian in the same units as R
    -  `flux`: Central surface density
    """
    return (flux / (sq_2pi * sigma)) * torch.exp(-0.5 * torch.pow(R / sigma, 2))
