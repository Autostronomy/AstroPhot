from ...backend_obj import backend, ArrayLike
import numpy as np


def gaussian(R: ArrayLike, sigma: ArrayLike, flux: ArrayLike) -> ArrayLike:
    """Gaussian 2d profile function.

    **Args:**
    -  `R`: Radii array at which to evaluate the gaussian function
    -  `sigma`: Standard deviation of the gaussian in the same units as R
    -  `flux`: Total flux of the Gaussian
    """
    return (flux / (2 * np.pi * sigma**2)) * backend.exp(-0.5 * (R / sigma) ** 2)
