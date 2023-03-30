import numpy as np
import logging
from scipy.stats import iqr

from .ellipse import parametric_SuperEllipse, Rscale_SuperEllipse
from ..conversions.coordinates import Rotate_Cartesian_np
from ..interpolate import interpolate_Lanczos


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


def _iso_between(
    IMG,
    sma_low,
    sma_high,
    PARAMS,
    c,
    more=False,
    mask=None,
    sigmaclip=False,
    sclip_iterations=10,
    sclip_nsigma=5,
):
    if not "m" in PARAMS:
        PARAMS["m"] = None
    if not "C" in PARAMS:
        PARAMS["C"] = None
    Rlim = sma_high * (
        1.0
        if PARAMS["m"] is None
        else np.exp(sum(np.abs(PARAMS["Am"][m]) for m in range(len(PARAMS["m"]))))
    )
    ranges = [
        [max(0, int(c["x"] - Rlim - 2)), min(IMG.shape[1], int(c["x"] + Rlim + 2))],
        [max(0, int(c["y"] - Rlim - 2)), min(IMG.shape[0], int(c["y"] + Rlim + 2))],
    ]
    XX, YY = np.meshgrid(
        np.arange(ranges[0][1] - ranges[0][0], dtype=float)
        - c["x"]
        + float(ranges[0][0]),
        np.arange(ranges[1][1] - ranges[1][0], dtype=float)
        - c["y"]
        + float(ranges[1][0]),
    )
    theta = np.arctan(YY / XX) + np.pi * (XX < 0)
    RR = np.sqrt(XX ** 2 + YY ** 2)
    Fmode_Rscale = (
        1.0
        if PARAMS["m"] is None
        else Rscale_Fmodes(
            theta - PARAMS["pa"], PARAMS["m"], PARAMS["Am"], PARAMS["Phim"]
        )
    )
    SuperEllipse_Rscale = Rscale_SuperEllipse(
        theta - PARAMS["pa"], PARAMS["ellip"], 2 if PARAMS["C"] is None else PARAMS["C"]
    )
    RR /= SuperEllipse_Rscale * Fmode_Rscale
    rselect = np.logical_and(RR < sma_high, RR > sma_low)
    fluxes = IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][rselect]
    CHOOSE = None
    if not mask is None and sma_high > 5:
        CHOOSE = np.logical_not(
            mask[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][rselect]
        )
    # Perform sigma clipping if requested
    if sigmaclip:
        sclim = Sigma_Clip_Upper(fluxes, sclip_iterations, sclip_nsigma)
        if CHOOSE is None:
            CHOOSE = fluxes < sclim
        else:
            CHOOSE = np.logical_or(CHOOSE, fluxes < sclim)
    if CHOOSE is not None and np.sum(CHOOSE) < 5:
        logging.warning(
            "Entire Isophote is Masked! R_l: %.3f, R_h: %.3f, PA: %.3f, ellip: %.3f"
            % (sma_low, sma_high, PARAMS["pa"] * 180 / np.pi, PARAMS["ellip"])
        )
        CHOOSE = np.ones(CHOOSE.shape).astype(bool)
    if CHOOSE is not None:
        countmasked = np.sum(np.logical_not(CHOOSE))
    else:
        countmasked = 0
    if more:
        if CHOOSE is not None and sma_high > 5:
            return fluxes[CHOOSE], theta[rselect][CHOOSE], countmasked
        else:
            return fluxes, theta[rselect], countmasked
    else:
        if CHOOSE is not None and sma_high > 5:
            return fluxes[CHOOSE]
        else:
            return fluxes


def _iso_extract(
    IMG,
    sma,
    PARAMS,
    c,
    more=False,
    minN=None,
    mask=None,
    interp_mask=False,
    rad_interp=30,
    interp_method="lanczos",
    interp_window=5,
    sigmaclip=False,
    sclip_iterations=10,
    sclip_nsigma=5,
):
    """
    Internal, basic function for extracting the pixel fluxes along an isophote
    """
    if not "m" in PARAMS:
        PARAMS["m"] = None
    if not "C" in PARAMS:
        PARAMS["C"] = None
    N = max(15, int(0.9 * 2 * np.pi * sma))
    if not minN is None:
        N = max(minN, N)
    # points along ellipse to evaluate
    theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
    theta = np.arctan(PARAMS["q"] * np.tan(theta)) + np.pi * (np.cos(theta) < 0)
    Fmode_Rscale = (
        1.0
        if PARAMS["m"] is None
        else Rscale_Fmodes(theta, PARAMS["m"], PARAMS["Am"], PARAMS["Phim"])
    )
    R = sma * Fmode_Rscale
    # Define ellipse
    X, Y = parametric_SuperEllipse(
        theta, 1.0 - PARAMS["q"], 2 if PARAMS["C"] is None else PARAMS["C"]
    )
    X, Y = R * X, R * Y
    # rotate ellipse by PA
    X, Y = Rotate_Cartesian_np(PARAMS["pa"], X, Y)
    theta = (theta + PARAMS["pa"]) % (2 * np.pi)
    # shift center
    X, Y = X + c["x"], Y + c["y"]

    # Reject samples from outside the image
    BORDER = np.logical_and(
        np.logical_and(X >= 0, X < (IMG.shape[1] - 1)),
        np.logical_and(Y >= 0, Y < (IMG.shape[0] - 1)),
    )
    if not np.all(BORDER):
        X = X[BORDER]
        Y = Y[BORDER]
        theta = theta[BORDER]

    Rlim = np.max(R)
    if Rlim < rad_interp:
        box = [
            [max(0, int(c["x"] - Rlim - 5)), min(IMG.shape[1], int(c["x"] + Rlim + 5))],
            [max(0, int(c["y"] - Rlim - 5)), min(IMG.shape[0], int(c["y"] + Rlim + 5))],
        ]
        if interp_method == "bicubic":
            flux = interpolate_bicubic(
                IMG[box[1][0] : box[1][1], box[0][0] : box[0][1]],
                X - box[0][0],
                Y - box[1][0],
            )
        elif interp_method == "lanczos":
            flux = interpolate_Lanczos(IMG, X, Y, interp_window)
        else:
            raise ValueError(
                "Unknown interpolate method %s. Should be one of lanczos or bicubic"
                % interp_method
            )
    else:
        # round to integers and sample pixels values
        flux = IMG[np.rint(Y).astype(np.int32), np.rint(X).astype(np.int32)]
    # CHOOSE holds bolean array for which flux values to keep, initialized as None for no clipping
    CHOOSE = None
    # Mask pixels if a mask is given
    if not mask is None:
        CHOOSE = np.logical_not(
            mask[np.rint(Y).astype(np.int32), np.rint(X).astype(np.int32)]
        )
    # Perform sigma clipping if requested
    if sigmaclip and len(flux) > 30:
        sclim = Sigma_Clip_Upper(flux, sclip_iterations, sclip_nsigma)
        if CHOOSE is None:
            CHOOSE = flux < sclim
        else:
            CHOOSE = np.logical_or(CHOOSE, flux < sclim)
    # Dont clip pixels if that removes all of the pixels
    countmasked = 0
    if not CHOOSE is None and np.sum(CHOOSE) <= 0:
        logging.warning(
            "Entire Isophote was Masked! R: %.3f, PA: %.3f, q: %.3f"
            % (sma, PARAMS["pa"] * 180 / np.pi, PARAMS["q"])
        )
        # Interpolate clipped flux values if requested
    elif not CHOOSE is None and interp_mask:
        flux[np.logical_not(CHOOSE)] = np.interp(
            theta[np.logical_not(CHOOSE)], theta[CHOOSE], flux[CHOOSE], period=2 * np.pi
        )
        # simply remove all clipped pixels if user doesn't reqest another option
    elif not CHOOSE is None:
        flux = flux[CHOOSE]
        theta = theta[CHOOSE]
        countmasked = np.sum(np.logical_not(CHOOSE))

    # Return just the flux values, or flux and angle values
    if more:
        return flux, theta, countmasked
    else:
        return flux


def _iso_line(IMG, length, width, pa, c, more=False):
    start = np.array([c["x"], c["y"]])
    end = start + length * np.array([np.cos(pa), np.sin(pa)])

    ranges = [
        [
            max(0, int(min(start[0], end[0]) - 2)),
            min(IMG.shape[1], int(max(start[0], end[0]) + 2)),
        ],
        [
            max(0, int(min(start[1], end[1]) - 2)),
            min(IMG.shape[0], int(max(start[1], end[1]) + 2)),
        ],
    ]
    XX, YY = np.meshgrid(
        np.arange(ranges[0][1] - ranges[0][0], dtype=float),
        np.arange(ranges[1][1] - ranges[1][0], dtype=float),
    )
    XX -= c["x"] - float(ranges[0][0])
    YY -= c["y"] - float(ranges[1][0])
    XX, YY = (XX * np.cos(-pa) - YY * np.sin(-pa), XX * np.sin(-pa) + YY * np.cos(-pa))

    lselect = np.logical_and.reduce(
        (XX >= -0.5, XX <= length, np.abs(YY) <= (width / 2))
    )
    flux = IMG[ranges[1][0] : ranges[1][1], ranges[0][0] : ranges[0][1]][lselect]

    if more:
        return flux, XX[lselect], YY[lselect]
    else:
        return flux, XX[lselect]
