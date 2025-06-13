import numpy as np
from scipy.optimize import minimize

from ..interpolate import point_Lanczos


def center_of_mass(image):
    """Determines the light weighted center of mass"""
    xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    center = np.array((np.sum(image * xx), np.sum(image * yy))) / np.sum(image)
    return center


def GaussianDensity_Peak(center, image, window=10, std=0.5):
    init_center = center
    window += window % 2

    def _add_flux(c):
        r = np.round(center)
        xx, yy = np.meshgrid(
            np.arange(r[0] - window / 2, r[0] + window / 2 + 1) - c[0],
            np.arange(r[1] - window / 2, r[1] + window / 2 + 1) - c[1],
        )
        rr2 = xx**2 + yy**2
        f = image[
            int(r[1] - window / 2) : int(r[1] + window / 2 + 1),
            int(r[0] - window / 2) : int(r[0] + window / 2 + 1),
        ]
        return -np.sum(np.exp(-rr2 / (2 * std)) * f)

    res = minimize(_add_flux, x0=center)
    return res.x


def Lanczos_peak(center, image, Lanczos_scale=3):
    best = [np.inf, None]
    for dx in np.arange(-3, 4):
        for dy in np.arange(-3, 4):
            res = minimize(
                lambda x: -point_Lanczos(image, x[0], x[1], scale=Lanczos_scale),
                x0=(center[0] + dx, center[1] + dy),
                method="Nelder-Mead",
            )
            if res.fun < best[0]:
                best[0] = res.fun
                best[1] = res.x
    return best[1]
