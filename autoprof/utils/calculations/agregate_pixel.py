import numpy as np
from scipy.stats import iqr


def Sigma_Clip_Upper(v, iterations=10, nsigma=5):
    """
    Perform sigma clipping on the "v" array. Each iteration involves
    computing the median and 16-84 range, these are used to clip beyond
    "nsigma" number of sigma above the median. This is repeated for
    "iterations" number of iterations, or until convergence if None.
    """

    v2 = np.sort(v)
    i = 0
    old_lim = 0
    lim = np.inf
    while i < iterations and old_lim != lim:
        med = np.median(v2[v2 < lim])
        rng = iqr(v2[v2 < lim], rng=[16, 84]) / 2
        old_lim = lim
        lim = med + rng * nsigma
        i += 1
    return lim


def Smooth_Mode(v):
    # set the starting point for the optimization at the median
    start = np.median(v)
    # set the smoothing scale equal to roughly 0.5% of the width of the data
    scale = iqr(v) / max(1.0, np.log10(len(v)))  # /10
    # Fit the peak of the smoothed histogram
    res = minimize(
        lambda x: -np.sum(np.exp(-(((v - x) / scale) ** 2))),
        x0=[start],
        method="Nelder-Mead",
    )
    return res.x[0]


def _average(v, method="median"):
    if method == "mean":
        return np.mean(v)
    elif method == "mode":
        return Smooth_Mode(v)
    elif method == "median":
        return np.median(v)
    else:
        raise ValueError("Unrecognized average method: %s" % method)


def _scatter(v, method="median"):
    if method == "mean":
        return np.std(v)
    elif method == "mode":
        return iqr(v, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0
    elif method == "median":
        return iqr(v, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0
    else:
        raise ValueError("Unrecognized average method: %s" % method)

