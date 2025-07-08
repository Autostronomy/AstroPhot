import numpy as np
from .conversions.functions import sersic_n_to_b


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
