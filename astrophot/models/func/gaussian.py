import torch
from ...backend_obj import backend, ArrayLike
import numpy as np

sq_2pi = np.sqrt(2 * np.pi)


def gaussian(R: ArrayLike, sigma: ArrayLike, flux: ArrayLike) -> ArrayLike:
    """Gaussian 1d profile function, specifically designed for pytorch
    operations.

    **Args:**
    -  `R`: Radii tensor at which to evaluate the gaussian function
    -  `sigma`: Standard deviation of the gaussian in the same units as R
    -  `flux`: Central surface density
    """
    return (flux / (sq_2pi * sigma)) * backend.exp(-0.5 * (R / sigma) ** 2)
