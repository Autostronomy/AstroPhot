import numpy as np


def Rscale_Fmodes(theta, modes, Am, Phim):
    return np.exp(
        sum(Am[m] * np.cos(modes[m] * (theta + Phim[m])) for m in range(len(modes)))
    )


def parametric_Fmodes(theta, modes, Am, Phim):
    x = np.cos(theta)
    y = np.sin(theta)
    Rscale = Rscale_Fmodes(theta, modes, Am, Phim)
    return x * Rscale, y * Rscale


def Rscale_SuperEllipse(theta, ellip, C=2):
    res = (1 - ellip) / (
        np.abs((1 - ellip) * np.cos(theta)) ** (C) + np.abs(np.sin(theta)) ** (C)
    ) ** (1.0 / C)
    return res

def parametric_SuperEllipse(theta, ellip, C=2):
    rs = Rscale_SuperEllipse(theta, ellip, C)
    return rs * np.cos(theta), rs * np.sin(theta)

