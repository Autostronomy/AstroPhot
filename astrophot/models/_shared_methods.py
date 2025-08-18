from scipy.stats import binned_statistic, iqr
import numpy as np
import torch
from scipy.optimize import minimize

from ..utils.decorators import ignore_numpy_warnings
from .. import config
from ..backend_obj import backend


def _sample_image(
    image,
    transform,
    radius,
    angle=None,
    rad_bins=None,
    angle_range=None,
    cycle=2 * np.pi,
):
    dat = backend.to_numpy(image.data).copy()
    # Fill masked pixels
    if image.has_mask:
        mask = backend.to_numpy(image.mask)
        dat[mask] = np.median(dat[~mask])
    # Subtract median of edge pixels to avoid effect of nearby sources
    edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
    dat -= np.median(edge)
    # Get the radius of each pixel relative to object center
    x, y = transform(*image.coordinate_center_meshgrid(), params=())
    R = backend.to_numpy(radius(x, y, params=())).flatten()

    if angle_range is not None:
        T = backend.to_numpy(angle(x, y, params=())).flatten()
        T = (T - angle_range[0]) % cycle
        CHOOSE = T < (angle_range[1] - angle_range[0])
        R = R[CHOOSE]
        dat = dat.flatten()[CHOOSE]
    raveldat = dat.ravel()

    # Bin fluxes by radius
    if rad_bins is None:
        rad_bins = np.logspace(
            np.log10(R.min() * 0.9 + image.pixelscale / 2), np.log10(R.max() * 1.1), 11
        )
    else:
        rad_bins = np.array(rad_bins)
    I = (
        binned_statistic(R, raveldat, statistic="median", bins=rad_bins)[0]
    ) / image.pixel_area.item()
    sigma = lambda d: iqr(d, rng=[16, 84]) / 2
    S = binned_statistic(R, raveldat, statistic=sigma, bins=rad_bins)[0] / image.pixel_area.item()
    R = (rad_bins[:-1] + rad_bins[1:]) / 2

    # Ensure enough values are positive
    N = np.isfinite(I)
    I[~N] = np.interp(R[~N], R[N], I[N])
    if np.sum(I > 0) <= 3:
        I = np.abs(I)
    N = I > 0
    if not np.all(N):
        I[~N] = np.interp(R[~N], R[N], I[N])
    # Ensure decreasing brightness with radius in outer regions
    for i in range(5, len(I)):
        if I[i] >= I[i - 1]:
            I[i] = I[i - 1] * 0.9
    # Convert to log scale
    S = S / (I * np.log(10))
    I = np.log10(I)
    # Ensure finite
    N = np.isfinite(I)
    if not np.all(N):
        I[~N] = np.interp(R[~N], R[N], I[N])
    N = np.isfinite(S)
    if not np.all(N):
        S[~N] = np.abs(np.interp(R[~N], R[N], S[N]))
    Sm = np.median(S)
    S[S < Sm] = Sm  # remove very small uncertainties

    return R, I, S


# General parametric
######################################################################
@torch.no_grad()
@ignore_numpy_warnings
def parametric_initialize(model, target, prof_func, params, x0_func):
    if all(list(model[param].initialized for param in params)):
        return
    # Get the sub-image area corresponding to the model image
    R, I, S = _sample_image(target, model.transform_coordinates, model.radius_metric)

    x0 = list(x0_func(model, R, I))
    for i, param in enumerate(params):
        x0[i] = x0[i] if not model[param].initialized else model[param].npvalue

    def optim(x, r, f, u):
        residual = ((f - np.nan_to_num(np.log10(prof_func(r, *x)), nan=np.min(f))) / u) ** 2
        N = np.argsort(residual)
        return np.mean(residual[N][:-2])

    res = minimize(optim, x0=x0, args=(R, I, S), method="Nelder-Mead")

    if res.success:
        x0 = res.x

    for param, x0x in zip(params, x0):
        if not model[param].initialized:
            if not model[param].is_valid(x0x):
                x0x = model[param].soft_valid(
                    backend.as_array(x0x, dtype=config.DTYPE, device=config.DEVICE)
                )
            model[param].dynamic_value = x0x


@torch.no_grad()
@ignore_numpy_warnings
def parametric_segment_initialize(
    model=None,
    target=None,
    prof_func=None,
    params=None,
    x0_func=None,
    segments=None,
):
    if all(list(model[param].initialized for param in params)):
        return

    cycle = np.pi if model.symmetric else 2 * np.pi
    w = cycle / segments
    v = w * np.arange(segments)
    values = []
    for s in range(segments):
        angle_range = (v[s] - w / 2, v[s] + w / 2)
        # Get the sub-image area corresponding to the model image
        R, I, S = _sample_image(
            target,
            model.transform_coordinates,
            model.radius_metric,
            angle=model.angular_metric,
            angle_range=angle_range,
            cycle=cycle,
        )

        x0 = list(x0_func(model, R, I))

        def optim(x, r, f, u):
            residual = ((f - np.log10(prof_func(r, *x))) / u) ** 2
            N = np.argsort(residual)
            return np.mean(residual[N][:-2])

        res = minimize(optim, x0=x0, args=(R, I, S), method="Nelder-Mead")
        if res.success:
            x0 = res.x

        values.append(x0)
    values = np.stack(values).T
    for param, v in zip(params, values):
        if not model[param].initialized:
            model[param].dynamic_value = v
