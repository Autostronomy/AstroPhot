import numpy as np
from numpy import ndarray
from .conversions.functions import sersic_n_to_b

__all__ = (
    "sersic_np",
    "gaussian_np",
    "exponential_np",
    "moffat_np",
    "nuker_np",
    "ferrer_np",
    "king_np",
)


def sersic_np(R: ndarray, n: ndarray, Re: ndarray, Ie: ndarray) -> ndarray:
    """Sersic 1d profile function, works more generally with numpy
    operations. In the event that impossible values are passed to the
    function it returns large values to guide optimizers away from
    such values.

    **Args:**
    -  `R`: Radii array at which to evaluate the sersic function
    -  `n`: sersic index restricted to n > 0.36
    -  `Re`: Effective radius in the same units as R
    -  `Ie`: Effective surface density
    """
    if np.any(np.array([n, Re, Ie]) <= 0):
        return np.ones(len(R)) * 1e6
    bn = sersic_n_to_b(n)
    return Ie * np.exp(-bn * ((R / Re) ** (1 / n) - 1))


def gaussian_np(R: ndarray, sigma: ndarray, I0: ndarray) -> ndarray:
    """Gaussian 1d profile function, works more generally with numpy
    operations.

    **Args:**
    -  `R`: Radii array at which to evaluate the gaussian function
    -  `sigma`: standard deviation of the gaussian in the same units as R
    -  `I0`: central surface density
    """
    return (I0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((R / sigma) ** 2))


def exponential_np(R: ndarray, Ie: ndarray, Re: ndarray) -> ndarray:
    """Exponential 1d profile function, works more generally with numpy
    operations.

    **Args:**
    -  `R`: Radii array at which to evaluate the exponential function
    -  `Ie`: Effective surface density
    -  `Re`: Effective radius in the same units as R
    """
    return Ie * np.exp(-sersic_n_to_b(1.0) * (R / Re - 1.0))


def moffat_np(R: ndarray, n: ndarray, Rd: ndarray, I0: ndarray) -> ndarray:
    """Moffat 1d profile function, works with numpy operations.

    **Args:**
    -  `R`: Radii array at which to evaluate the moffat function
    -  `n`: concentration index
    -  `Rd`: scale length in the same units as R
    -  `I0`: central surface density
    """
    return I0 / (1 + (R / Rd) ** 2) ** n


def nuker_np(
    R: ndarray, Rb: ndarray, Ib: ndarray, alpha: ndarray, beta: ndarray, gamma: ndarray
) -> ndarray:
    """Nuker 1d profile function, works with numpy functions

    **Args:**
    -  `R`: Radii tensor at which to evaluate the nuker function
    -  `Ib`: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
    -  `Rb`: scale length radius
    -  `alpha`: sharpness of transition between power law slopes
    -  `beta`: outer power law slope
    -  `gamma`: inner power law slope

    """
    return (
        Ib
        * (2 ** ((beta - gamma) / alpha))
        * ((R / Rb) ** (-gamma))
        * ((1 + (R / Rb) ** alpha) ** ((gamma - beta) / alpha))
    )


def ferrer_np(R: ndarray, rout: ndarray, alpha: ndarray, beta: ndarray, I0: ndarray) -> ndarray:
    """
    Modified Ferrer profile.

    **Args:**
    -  `R`: Radial distance from the center.
    -  `rout`: Outer radius of the profile.
    -  `alpha`: Power-law index.
    -  `beta`: Exponent for the modified Ferrer function.
    -  `I0`: Central intensity.
    """
    return (R < rout) * I0 * ((1 - (np.clip(R, 0, rout) / rout) ** (2 - beta)) ** alpha)


def king_np(R: ndarray, Rc: ndarray, Rt: ndarray, alpha: ndarray, I0: ndarray) -> ndarray:
    """
    Empirical King profile.

    **Args:**
    -  `R`: The radial distance from the center.
    -  `Rc`: The core radius of the profile.
    -  `Rt`: The truncation radius of the profile.
    -  `alpha`: The power-law index of the profile.
    -  `I0`: The central intensity of the profile.
    """
    beta = 1 / (1 + (Rt / Rc) ** 2) ** (1 / alpha)
    gamma = 1 / (1 + (R / Rc) ** 2) ** (1 / alpha)
    return (R < Rt) * I0 * ((np.clip(gamma, 0, 1) - beta) / (1 - beta)) ** alpha
