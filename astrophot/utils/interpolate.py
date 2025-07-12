import torch
import numpy as np


def default_prof(shape, pixelscale, min_pixels=2, scale=0.2):
    prof = [0, min_pixels * pixelscale]
    while prof[-1] < (np.max(shape) * pixelscale / 2):
        prof.append(prof[-1] + max(min_pixels * pixelscale, prof[-1] * scale))
    return prof


def interp1d_torch(x_in, y_in, x_out):
    indices = torch.searchsorted(x_in[:-1], x_out) - 1
    weights = (y_in[1:] - y_in[:-1]) / (x_in[1:] - x_in[:-1])
    return y_in[indices] + weights[indices] * (x_out - x_in[indices])


def interp2d(
    im: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates a 2D image at specified coordinates.
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Args:
        im (Tensor): A 2D tensor representing the image.
        x (Tensor): A tensor of x coordinates (in pixel space) at which to interpolate.
        y (Tensor): A tensor of y coordinates (in pixel space) at which to interpolate.

    Returns:
        Tensor: Tensor with the same shape as `x` and `y` containing the interpolated values.
    """

    # Convert coordinates to pixel indices
    h, w = im.shape

    # reshape for indexing purposes
    start_shape = x.shape
    x = x.flatten()
    y = y.flatten()

    # valid
    valid = (x >= -0.5) & (x <= (w - 0.5)) & (y >= -0.5) & (y <= (h - 0.5))

    x0 = x.floor().long()
    y0 = y.floor().long()
    x0 = x0.clamp(0, w - 2)
    x1 = x0 + 1
    y0 = y0.clamp(0, h - 2)
    y1 = y0 + 1

    fa = im[y0, x0]
    fb = im[y1, x0]
    fc = im[y0, x1]
    fd = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    result = fa * wa + fb * wb + fc * wc + fd * wd

    return (result * valid).reshape(start_shape)


def interp2d_ij(
    im: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates a 2D image at specified coordinates.
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Args:
        im (Tensor): A 2D tensor representing the image.
        x (Tensor): A tensor of x coordinates (in pixel space) at which to interpolate.
        y (Tensor): A tensor of y coordinates (in pixel space) at which to interpolate.

    Returns:
        Tensor: Tensor with the same shape as `x` and `y` containing the interpolated values.
    """

    # Convert coordinates to pixel indices
    h, w = im.shape

    # reshape for indexing purposes
    start_shape = i.shape
    i = i.flatten()
    j = j.flatten()

    # valid
    valid = (i >= -0.5) & (i <= (h - 0.5)) & (j >= -0.5) & (j <= (w - 0.5))

    i0 = i.floor().long()
    j0 = j.floor().long()
    i0 = i0.clamp(0, h - 2)
    i1 = i0 + 1
    j0 = j0.clamp(0, w - 2)
    j1 = j0 + 1

    fa = im[i0, j0]
    fb = im[i0, j1]
    fc = im[i1, j0]
    fd = im[i1, j1]

    wa = (i1 - i) * (j1 - j)
    wb = (i1 - i) * (j - j0)
    wc = (i - i0) * (j1 - j)
    wd = (i - i0) * (j - j0)

    result = fa * wa + fb * wb + fc * wc + fd * wd

    return (result * valid).view(*start_shape)
