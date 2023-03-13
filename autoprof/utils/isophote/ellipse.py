import numpy as np


def Rscale_Fmodes(theta, modes, Am, Phim):
    """Factor to scale radius values given a set of fourier mode
    amplitudes.

    """
    return np.exp(
        sum(Am[m] * np.cos(modes[m] * (theta + Phim[m])) for m in range(len(modes)))
    )


def parametric_Fmodes(theta, modes, Am, Phim):
    """determines a number of scaled radius samples with fourier mode
    perturbations for a unit circle.

    """
    x = np.cos(theta)
    y = np.sin(theta)
    Rscale = Rscale_Fmodes(theta, modes, Am, Phim)
    return x * Rscale, y * Rscale


def Rscale_SuperEllipse(theta, ellip, C=2):
    """Scale factor for radius values given a super ellipse coefficient."""
    res = (1 - ellip) / (
        np.abs((1 - ellip) * np.cos(theta)) ** (C) + np.abs(np.sin(theta)) ** (C)
    ) ** (1.0 / C)
    return res


def parametric_SuperEllipse(theta, ellip, C=2):
    """determines a number of scaled radius samples with super ellipse
    perturbations for a unit circle.

    """
    rs = Rscale_SuperEllipse(theta, ellip, C)
    return rs * np.cos(theta), rs * np.sin(theta)
