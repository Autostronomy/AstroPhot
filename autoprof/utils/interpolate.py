import numpy as np
from astropy.convolution import convolve, convolve_fft
import torch
from torch.nn.functional import conv2d

def window_function(img, X, Y, func, window):
    pass


import torch

def _h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt
def cubic_spline_torch(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = _h_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx


def interpolate_bicubic(img, X, Y):
    f_interp = RectBivariateSpline(
        np.arange(dat.shape[0], dtype=np.float32),
        np.arange(dat.shape[1], dtype=np.float32),
        dat,
    )
    return f_interp(Y, X, grid=False)

def Lanczos_kernel(dx, dy, scale):
    xx = np.arange(int(-scale+1), int(scale+1)) + dx
    yy = np.arange(int(-scale+1), int(scale+1)) + dy
    Lx = np.sinc(xx) * np.sinc(xx / scale)
    Ly = np.sinc(yy) * np.sinc(yy / scale)
    LXX, LYY = np.meshgrid(Lx, Ly)
    LL = LXX * LYY
    w = np.sum(LL)
    LL /= w
    return LL
    
def point_Lanczos(I, X, Y, scale):
    """
    Apply Lanczos interpolation to evaluate a single point
    """
    ranges = [
        [int(np.floor(X)-scale), int(np.floor(X)+scale)],
        [int(np.floor(Y)-scale), int(np.floor(Y)+scale)],
    ]
    LL = Lanczos_kernel(np.floor(X) - X, np.floor(Y) - Y, scale)
    LL = LL[
        max(0,-ranges[1][0]):LL.shape[0] + min(0,I.shape[0] - ranges[1][1]),
        max(0,-ranges[0][0]):LL.shape[1] + min(0,I.shape[1] - ranges[0][1]),
    ]
    F = I[
        max(0,ranges[1][0]):min(I.shape[0],ranges[1][1]),
        max(0,ranges[0][0]):min(I.shape[1],ranges[0][1]),
    ]
    return np.sum(F * LL)

def arbitrary_Lanczos(I, X, Y, scale):
    """
    Apply Lanczos interpolation for a list of coordinates with unspecified structure.
    """
    F = []
    for x, y in zip(X, Y):
        F.append(point_Lanczos(I, x, y, scale))
    return np.array(F)

def _shift_Lanczos_kernel(dx, dy, scale):
    xx = torch.arange(int(-scale), int(scale+1)) + dx
    yy = torch.arange(int(-scale), int(scale+1)) + dy
    Lx = torch.sinc(xx) * torch.sinc(xx / scale)
    Ly = torch.sinc(yy) * torch.sinc(yy / scale)
    Lx[0] = 0
    Ly[0] = 0
    LXX, LYY = torch.meshgrid(Lx, Ly, indexing = 'xy')
    LL = LXX * LYY
    w = torch.sum(LL)
    LL /= w
    return LL

def shift_Lanczos(I, dx, dy, scale):
    """
    Apply Lanczos interpolation to shift by less than a pixel in x and y
    """
    LL = _shift_Lanczos_kernel(-dx,-dy, scale)
    return conv2d(I.view(1,1,*I.shape), LL.view(1,1,*LL.shape), padding = "same")[0][0]

def interpolate_Lanczos_grid(img, X, Y, scale):
    """
    Perform Lanczos interpolation at a grid of points.
    https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html
    """
    
    sinc_X = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i]))
        * np.sinc(
            (np.arange(-scale + 1, scale + 1) - X[i] + np.floor(X[i])) / scale
        )
        for i in range(len(X))
    )
    sinc_Y = list(
        np.sinc(np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i]))
        * np.sinc(
            (np.arange(-scale + 1, scale + 1) - Y[i] + np.floor(Y[i])) / scale
        )
        for i in range(len(Y))
    )

    # Extract an image which has the required dimensions
    use_img = np.take(
        np.take(img, np.arange(int(np.floor(Y[0]) - step + 1), int(np.floor(Y[-1]) + step + 1)), 0, mode = "clip"),
        np.arange(int(np.floor(X[0]) - step + 1), int(np.floor(X[-1]) + step + 1)), 1, mode = "clip"
    )

    # Create a sliding window view of the image with the dimensions of the lanczos scale grid
    #window = np.lib.stride_tricks.sliding_window_view(use_img, (2*scale, 2*scale))

    # fixme going to need some broadcasting magic
    XX = np.ones((2*scale,2*scale))
    res = np.zeros((len(Y), len(X)))
    for x, lowx, highx in zip(range(len(X)), np.floor(X) - step + 1, np.floor(X) + step + 1):
        for y, lowy, highy in zip(range(len(Y)), np.floor(Y) - step + 1, np.floor(Y) + step + 1):
            L = XX * sinc_X[x] * sinc_Y[y].reshape((sinc_Y[y].size, -1))
            res[y,x] = np.sum(use_img[lowy:highy,lowx:highx] * L) / np.sum(L)
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

def nearest_neighbor(img, X, Y):
    return img[
        np.clip(np.round(Y).astype(int), a_min = 0, a_max = img.shape[0] - 1),
        np.clip(np.round(X).astype(int), a_min = 0, a_max = img.shape[1] - 1),
    ]

