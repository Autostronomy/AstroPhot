import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

from ..interpolate import point_Lanczos
from ... import AP_config


def center_of_mass(center, image, window=None):
    """Iterative light weighted center of mass optimization. Each step
    determines the light weighted center of mass within a small
    window. The new center is used to create a new window. This
    continues until the center no longer updates or an image boundary
    is reached.

    """
    if window is None:
        window = max(min(int(min(image.shape) / 10), 30), 6)
    init_center = center
    window += window % 2
    xx, yy = np.meshgrid(np.arange(window), np.arange(window))
    for iteration in range(100):
        # Determine the image window to calculate COM
        ranges = [
            [int(round(center[0]) - window / 2), int(round(center[0]) + window / 2)],
            [int(round(center[1]) - window / 2), int(round(center[1]) + window / 2)],
        ]
        # Avoid edge of image
        if (
            ranges[0][0] < 0
            or ranges[1][0] < 0
            or ranges[0][1] >= image.shape[0]
            or ranges[1][1] >= image.shape[1]
        ):
            AP_config.ap_logger.warning("Image edge!")
            return init_center

        # Compute COM
        denom = np.sum(image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]])
        new_center = [
            ranges[0][0]
            + np.sum(
                image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]] * yy
            )
            / denom,
            ranges[1][0]
            + np.sum(
                image[ranges[0][0] : ranges[0][1], ranges[1][0] : ranges[1][1]] * xx
            )
            / denom,
        ]
        new_center = np.array(new_center)
        # Check for convergence
        if np.sum(np.abs(np.array(center) - new_center)) < 0.1:
            break

        center = new_center

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
        rr2 = xx ** 2 + yy ** 2
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
