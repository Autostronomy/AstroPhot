from functools import lru_cache

import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft
from torch.nn.functional import conv2d

from .operations import fft_convolve_torch


def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Args:
        t (Tensor): A 1D tensor representing the normalized x values.

    Returns:
        Tensor: A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (torch.arange(4, device=t.device)[:, None])
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt


def cubic_spline_torch(
    x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor, extend: str = "const"
) -> torch.Tensor:
    """Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    Args:
        x (Tensor): A 1D tensor representing the x-coordinates of the known data points.
        y (Tensor): A 1D tensor representing the y-coordinates of the known data points.
        xs (Tensor): A 1D tensor representing the x-coordinates of the positions where
                     the cubic spline function should be evaluated.
        extend (str, optional): The method for handling extrapolation, either "const" or "linear".
                                Default is "const".
                                "const": Use the value of the last known data point for extrapolation.
                                "linear": Use linear extrapolation based on the last two known data points.

    Returns:
        Tensor: A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = (
        hh[0] * y[idxs]
        + hh[1] * m[idxs] * dx
        + hh[2] * y[idxs + 1]
        + hh[3] * m[idxs + 1] * dx
    )
    if extend == "const":
        ret[xs > x[-1]] = y[-1]
    elif extend == "linear":
        indices = xs > x[-1]
        ret[indices] = y[-1] + (xs[indices] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return ret


def interpolate_bicubic(img, X, Y):
    """
    wrapper for scipy bivariate spline interpolation
    """
    f_interp = RectBivariateSpline(
        np.arange(dat.shape[0], dtype=np.float32),
        np.arange(dat.shape[1], dtype=np.float32),
        dat,
    )
    return f_interp(Y, X, grid=False)


def Lanczos_kernel_np(dx, dy, scale):
    """convolution kernel for shifting all pixels in a grid by some
    sub-pixel length.

    """
    xx = np.arange(-scale, scale + 1) - dx
    if dx < 0:
        xx *= -1
    Lx = np.sinc(xx) * np.sinc(xx / scale)
    if dx > 0:
        Lx[0] = 0
    else:
        Lx[-1] = 0

    yy = np.arange(-scale, scale + 1) - dy
    if dy < 0:
        yy *= -1
    Ly = np.sinc(yy) * np.sinc(yy / scale)
    if dx > 0:
        Ly[0] = 0
    else:
        Ly[-1] = 0

    LXX, LYY = np.meshgrid(Lx, Ly, indexing="xy")
    LL = LXX * LYY
    w = np.sum(LL)
    LL /= w
    # plt.imshow(LL.detach().numpy(), origin = "lower")
    # plt.show()
    return LL


def Lanczos_kernel(dx, dy, scale):
    """Kernel function for Lanczos interpolation, defines the
    interpolation behavior between pixels.

    """
    xx = np.arange(-scale + 1, scale + 1) + dx
    yy = np.arange(-scale + 1, scale + 1) + dy
    Lx = np.sinc(xx) * np.sinc(xx / scale)
    Ly = np.sinc(yy) * np.sinc(yy / scale)
    LXX, LYY = np.meshgrid(Lx, Ly)
    LL = LXX * LYY
    w = np.sum(LL)
    LL /= w
    return LL


def point_Lanczos(I, X, Y, scale):
    """
    Apply Lanczos interpolation to evaluate a single point.
    """
    ranges = [
        [int(np.floor(X) - scale + 1), int(np.floor(X) + scale + 1)],
        [int(np.floor(Y) - scale + 1), int(np.floor(Y) + scale + 1)],
    ]
    LL = Lanczos_kernel(np.floor(X) - X, np.floor(Y) - Y, scale)
    LL = LL[
        max(0, -ranges[1][0]) : LL.shape[0] + min(0, I.shape[0] - ranges[1][1]),
        max(0, -ranges[0][0]) : LL.shape[1] + min(0, I.shape[1] - ranges[0][1]),
    ]
    F = I[
        max(0, ranges[1][0]) : min(I.shape[0], ranges[1][1]),
        max(0, ranges[0][0]) : min(I.shape[1], ranges[0][1]),
    ]
    return np.sum(F * LL)


def _shift_Lanczos_kernel_torch(dx, dy, scale, dtype, device):
    """convolution kernel for shifting all pixels in a grid by some
    sub-pixel length.

    """
    xsign = 1 - 2 * (dx < 0).to(
        dtype=torch.int32
    )  # flips the kernel if the shift is negative
    xx = xsign * (
        torch.arange(int(-scale), int(scale + 1), dtype=dtype, device=device) - dx
    )
    Lx = torch.sinc(xx) * torch.sinc(xx / scale)

    ysign = 1 - 2 * (dy < 0).to(dtype=torch.int32)
    yy = ysign * (
        torch.arange(int(-scale), int(scale + 1), dtype=dtype, device=device) - dy
    )
    Ly = torch.sinc(yy) * torch.sinc(yy / scale)

    LXX, LYY = torch.meshgrid(Lx, Ly, indexing="xy")
    LL = LXX * LYY
    w = torch.sum(LL)
    # plt.imshow(LL.detach().numpy(), origin = "lower")
    # plt.show()
    return LL / w


def shift_Lanczos_torch(I, dx, dy, scale, dtype, device, img_prepadded=True):
    """Apply Lanczos interpolation to shift by less than a pixel in x and
    y.

    """
    LL = _shift_Lanczos_kernel_torch(dx, dy, scale, dtype, device)
    ret = fft_convolve_torch(I, LL, img_prepadded=img_prepadded)
    return ret


def shift_Lanczos_np(I, dx, dy, scale):
    """Apply Lanczos interpolation to shift by less than a pixel in x and
    y.

    I: the image
    dx: amount by which the grid will be moved in the x-axis (the "data" is fixed and the grid moves). Should be a value from (-0.5,0.5)
    dy: amount by which the grid will be moved in the y-axis (the "data" is fixed and the grid moves). Should be a value from (-0.5,0.5)
    scale: dictates size of the Lanczos kernel. Full kernel size is 2*scale+1
    """
    LL = Lanczos_kernel_np(dx, dy, scale)
    return convolve_fft(I, LL, boundary="fill")


def interpolate_Lanczos_grid(img, X, Y, scale):
    """
    Perform Lanczos interpolation at a grid of points.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """

    sinc_X = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
        * np.sinc((np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale)
        for i in range(len(X))
    )
    sinc_Y = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
        * np.sinc((np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale)
        for i in range(len(Y))
    )

    # Extract an image which has the required dimensions
    use_img = np.take(
        np.take(
            img,
            np.arange(int(np.floor(Y[0]) - step + 1), int(np.floor(Y[-1]) + step + 1)),
            0,
            mode="clip",
        ),
        np.arange(int(np.floor(X[0]) - step + 1), int(np.floor(X[-1]) + step + 1)),
        1,
        mode="clip",
    )

    # Create a sliding window view of the image with the dimensions of the lanczos scale grid
    # window = np.lib.stride_tricks.sliding_window_view(use_img, (2*scale, 2*scale))

    # fixme going to need some broadcasting magic
    XX = np.ones((2 * scale, 2 * scale))
    res = np.zeros((len(Y), len(X)))
    for x, lowx, highx in zip(
        range(len(X)), np.floor(X) - step + 1, np.floor(X) + step + 1
    ):
        for y, lowy, highy in zip(
            range(len(Y)), np.floor(Y) - step + 1, np.floor(Y) + step + 1
        ):
            L = XX * sinc_X[x] * sinc_Y[y].reshape((sinc_Y[y].size, -1))
            res[y, x] = np.sum(use_img[lowy:highy, lowx:highx] * L) / np.sum(L)
    return res


def interpolate_Lanczos(img, X, Y, scale):
    """
    Perform Lanczos interpolation on an image at a series of specified points.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """
    flux = []

    for i in range(len(X)):
        box = [
            [
                max(0, int(round(np.floor(X[i]) - scale + 1))),
                min(img.shape[1], int(round(np.floor(X[i]) + scale + 1))),
            ],
            [
                max(0, int(round(np.floor(Y[i]) - scale + 1))),
                min(img.shape[0], int(round(np.floor(Y[i]) + scale + 1))),
            ],
        ]
        chunk = img[box[1][0] : box[1][1], box[0][0] : box[0][1]]
        XX = np.ones(chunk.shape)
        Lx = (
            np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale
            )
        )[
            box[0][0]
            - int(round(np.floor(X[i]) - scale + 1)) : 2 * scale
            + box[0][1]
            - int(round(np.floor(X[i]) + scale + 1))
        ]
        Ly = (
            np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
            * np.sinc(
                (np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale
            )
        )[
            box[1][0]
            - int(round(np.floor(Y[i]) - scale + 1)) : 2 * scale
            + box[1][1]
            - int(round(np.floor(Y[i]) + scale + 1))
        ]
        L = XX * Lx * Ly.reshape((Ly.size, -1))
        w = np.sum(L)
        flux.append(np.sum(chunk * L) / w)
    return np.array(flux)


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
    x = x.view(-1)
    y = y.view(-1)

    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = x0.clamp(0, w - 2)
    x1 = x1.clamp(1, w - 1)
    y0 = y0.clamp(0, h - 2)
    y1 = y1.clamp(1, h - 1)

    fa = im[y0, x0]
    fb = im[y1, x0]
    fc = im[y0, x1]
    fd = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    result = fa * wa + fb * wb + fc * wc + fd * wd

    return result.view(*start_shape)


@lru_cache(maxsize=32)
def curvature_kernel(dtype, device):
    kernel = (
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4, 1.0],
                [0.0, 1.0, 0.0],
            ],  # [[1., -2.0, 1.], [-2.0, 4, -2.0], [1.0, -2.0, 1.0]],
            device=device,
            dtype=dtype,
        )
    )
    return kernel


@lru_cache(maxsize=32)
def simpsons_kernel(dtype, device):
    kernel = torch.ones(1, 1, 3, 3, dtype=dtype, device=device)
    kernel[0, 0, 1, 1] = 16.0
    kernel[0, 0, 1, 0] = 4.0
    kernel[0, 0, 0, 1] = 4.0
    kernel[0, 0, 1, 2] = 4.0
    kernel[0, 0, 2, 1] = 4.0
    kernel = kernel / 36.0
    return kernel
