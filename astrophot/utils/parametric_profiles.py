import numpy as np
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
        return np.ones(len(R)) * 1e6
    bn = sersic_n_to_b(n)
    return Ie * np.exp(-bn * ((R / Re) ** (1 / n) - 1))


def gaussian_np(R, sigma, I0):
    """Gaussian 1d profile function, works more generally with numpy
    operations.

    Parameters:
        R: Radii array at which to evaluate the sersic function
        sigma: standard deviation of the gaussian in the same units as R
        I0: central surface density
    """
    return (I0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((R / sigma) ** 2))


def exponential_np(R, Ie, Re):
    """Exponential 1d profile function, works more generally with numpy
    operations.

    Parameters:
        R: Radii array at which to evaluate the sersic function
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    return Ie * np.exp(-sersic_n_to_b(1.0) * (R / Re - 1.0))


def moffat_np(R, n, Rd, I0):
    """Moffat 1d profile function, works with numpy operations.

    Parameters:
        R: Radii tensor at which to evaluate the moffat function
        n: concentration index
        Rd: scale length in the same units as R
        I0: central surface density

    """
    return I0 / (1 + (R / Rd) ** 2) ** n


def nuker_np(R, Rb, Ib, alpha, beta, gamma):
    """Nuker 1d profile function, works with numpy functions

    Parameters:
        R: Radii tensor at which to evaluate the nuker function
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """
    return (
        Ib
        * (2 ** ((beta - gamma) / alpha))
        * ((R / Rb) ** (-gamma))
        * ((1 + (R / Rb) ** alpha) ** ((gamma - beta) / alpha))
    )


def ferrer_np(R, rout, alpha, beta, I0):
    """
    Modified Ferrer profile.

    Parameters
    ----------
    R : array_like
        Radial distance from the center.
    rout : float
        Outer radius of the profile.
    alpha : float
        Power-law index.
    beta : float
        Exponent for the modified Ferrer function.
    I0 : float
        Central intensity.

    Returns
    -------
    array_like
        The modified Ferrer profile evaluated at R.
    """
    return (R < rout) * I0 * ((1 - (np.clip(R, 0, rout) / rout) ** (2 - beta)) ** alpha)


def king_np(R, Rc, Rt, alpha, I0):
    """
    Empirical King profile.

    Parameters
    ----------
    R : array_like
        The radial distance from the center.
    Rc : float
        The core radius of the profile.
    Rt : float
        The truncation radius of the profile.
    alpha : float
        The power-law index of the profile.
    I0 : float
        The central intensity of the profile.

    Returns
    -------
    array_like
        The intensity at each radial distance.
    """
    beta = 1 / (1 + (Rt / Rc) ** 2) ** (1 / alpha)
    gamma = 1 / (1 + (R / Rc) ** 2) ** (1 / alpha)
    return (R < Rt) * I0 * ((np.clip(gamma, 0, 1) - beta) / (1 - beta)) ** alpha
