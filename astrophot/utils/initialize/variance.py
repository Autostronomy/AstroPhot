import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
import torch
from ...errors import InvalidData
import matplotlib.pyplot as plt


def auto_variance(data, mask=None):

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask is None:
        mask = np.zeros(data.shape, dtype=int)

    # Data too small for anything fancy
    var = np.var(data[np.logical_not(mask)])
    if not np.isfinite(var) or var == 0:
        return np.ones_like(data)
    if min(data.shape) < 20:
        return np.ones_like(data) * var

    # Get approximation of noise in each pixel
    blur_data = gaussian_filter(data, 1.1)[4:-4, 4:-4]
    blur_mask = gaussian_filter(mask, 1.1)[4:-4, 4:-4] == 0
    mask = np.logical_not(mask)
    residuals = (data[4:-4, 4:-4] - blur_data)[blur_mask]

    # Bin the residuals by flux, clip the tails
    clips = (np.quantile(data[mask], 0.01), np.quantile(data[mask], 0.99))
    bins = np.linspace(clips[0], clips[1], 11)
    std, bins, _ = binned_statistic(
        data[4:-4, 4:-4][blur_mask].flatten(),
        residuals.flatten(),
        statistic="std",
        bins=bins,
    )
    N = np.logical_not(np.isfinite(std))
    if np.any(N):
        std[N] = np.sqrt(np.interp(bins[:-1][N], bins[:-1][~N], std[~N] ** 2))

    # Fit a line to the variance
    p = np.polyfit(bins[:-3], std[:-2] ** 2, 1)

    # Check if the variance is increasing with flux
    if p[0] < 0:
        raise InvalidData(
            "Variance appears to be decreasing with flux! Cannot accurately estimate variance."
        )
    # Compute the approximate variance map
    variance = np.clip(p[0] * data + p[1], np.min(std) ** 2, None)
    variance[np.logical_not(mask)] = np.inf
    return variance
