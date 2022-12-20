from ..isophote.extract import _iso_extract
import numpy as np
from scipy.stats import iqr
from scipy.fftpack import fft

def isophotes(image, center, threshold = None, pa = None, q = None, R = None, n_isophotes = 3, more = False):
    """Method for quickly extracting a small number of elliptical
    isophotes for the sake of initializing other models.

    """

    if pa is None:
        pa = 0.

    if q is None:
        q = 1.

    if R is None:
        # Determine basic threshold if none given
        if threshold is None:
            threshold = np.median(image) + 3*iqr(image, rng = (16,84))/2

        # Sample growing isophotes until threshold is reached
        ellipse_radii = [1.0]
        while ellipse_radii[-1] < (len(image) / 2):
            ellipse_radii.append(ellipse_radii[-1] * (1 + 0.2))
            isovals = _iso_extract(
                image,
                ellipse_radii[-1],
                {"q": q if isinstance(q, float) else np.max(q),
                 "pa": pa if isinstance(pa, float) else np.min(pa)},
                {"x": center[0], "y": center[1]},
                more=False,
                sigmaclip=True,
                sclip_nsigma=3,
            )
            if len(isovals) < 3:
                continue
            # Stop when at 3 time background noise
            if (np.quantile(isovals, 0.8) < threshold) and len(ellipse_radii) > 4:
                break
        R = ellipse_radii[-1]

    # Determine which radii to sample based on input R, pa, and q
    if isinstance(pa, float) and isinstance(q, float) and isinstance(R, float):
        if n_isophotes == 1:
            isophote_radii = [R]
        else:
            isophote_radii = np.linspace(0,R, n_isophotes)
    elif hasattr(R, "__len__"):
        isophote_radii = R
    elif hasattr(pa, "__len__"):
        isophote_radii = np.ones(len(pa)) * R
    elif hasattr(q, "__len__"):
        isophote_radii = np.ones(len(q)) * R
        
    # Sample the requested isophotes and record desired info
    iso_info = []
    for i, r in enumerate(isophote_radii):
        iso_info.append({"R": r})
        isovals = _iso_extract(
            image,
            r,
            {"q": q if isinstance(q, float) else q[i],
             "pa": pa if isinstance(pa, float) else pa[i]},
            {"x": center[0], "y": center[1]},
            more=more,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
        if more:
            angles = isovals[1]
            isovals = isovals[0]
        coefs = fft(isovals)
        iso_info[-1]["phase1"] = np.angle(coefs[1])
        iso_info[-1]["phase2"] = np.angle(coefs[2])
        iso_info[-1]["flux"] = np.median(isovals)
        iso_info[-1]["noise"] = iqr(isovals, rng = (16,84))/2
        iso_info[-1]["amplitude1"] = np.abs(coefs[1]) / (len(isovals) * (max(0,iso_info[-1]["flux"]) + iso_info[-1]["noise"]))
        iso_info[-1]["amplitude2"] = np.abs(coefs[2]) / (len(isovals) * (max(0,iso_info[-1]["flux"]) + iso_info[-1]["noise"]))
        iso_info[-1]["N"] = len(isovals)        
        if more:
            iso_info[-1]["isovals"] = isovals
            iso_info[-1]["angles"] = angles
    return iso_info
