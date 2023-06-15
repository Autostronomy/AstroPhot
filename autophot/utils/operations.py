from functools import lru_cache
from typing import Callable, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, convolve_fft
from scipy.fft import next_fast_len
from scipy.special import roots_legendre

def fft_convolve_torch(img, psf, psf_fft=False, img_prepadded=False):
    # Ensure everything is tensor
    img = torch.as_tensor(img)
    psf = torch.as_tensor(psf)

    if img_prepadded:
        s = img.size()
    else:
        s = tuple(
            next_fast_len(int(d + (p + 1) / 2), real=True)
            for d, p in zip(img.size(), psf.size())
        )  # list(int(d + (p + 1) / 2) for d, p in zip(img.size(), psf.size()))

    img_f = torch.fft.rfft2(img, s=s)

    if not psf_fft:
        psf_f = torch.fft.rfft2(psf, s=s)
    else:
        psf_f = psf

    conv_f = img_f * psf_f
    conv = torch.fft.irfft2(conv_f, s=s)

    # Roll the tensor to correct centering and crop to original image size
    return torch.roll(
        conv,
        shifts=(-int((psf.size()[0] - 1) / 2), -int((psf.size()[1] - 1) / 2)),
        dims=(0, 1),
    )[: img.size()[0], : img.size()[1]]


def fft_convolve_multi_torch(
    img, kernels, kernel_fft=False, img_prepadded=False, dtype=None, device=None
):
    # Ensure everything is tensor
    img = torch.as_tensor(img, dtype=dtype, device=device)
    for k in range(len(kernels)):
        kernels[k] = torch.as_tensor(kernels[k], dtype=dtype, device=device)

    if img_prepadded:
        s = img.size()
    else:
        s = list(int(d + (p + 1) / 2) for d, p in zip(img.size(), kernels[0].size()))

    img_f = torch.fft.rfft2(img, s=s)

    if not kernel_fft:
        kernels_f = list(torch.fft.rfft2(kernel, s=s) for kernel in kernels)
    else:
        psf_f = psf

    conv_f = img_f

    for kernel_f in kernels_f:
        conv_f *= kernel_f

    conv = torch.fft.irfft2(conv_f, s=s)

    # Roll the tensor to correct centering and crop to original image size
    return torch.roll(
        conv,
        shifts=(
            -int((sum(kernel.size()[0] for kernel in kernels) - 1) / 2),
            -int((sum(kernel.size()[1] for kernel in kernels) - 1) / 2),
        ),
        dims=(0, 1),
    )[: img.size()[0], : img.size()[1]]


def displacement_spacing(N, dtype=torch.float64, device="cpu"):
    return torch.linspace(
        -(N - 1) / (2 * N), (N - 1) / (2 * N), N, dtype=dtype, device=device
    )


def displacement_grid(Nx, Ny, pixelscale=None, dtype=torch.float64, device="cpu"):
    px = displacement_spacing(Nx, dtype=dtype, device=device)
    py = displacement_spacing(Ny, dtype=dtype, device=device)
    PX, PY = torch.meshgrid(px, py, indexing = "xy")
    return (pixelscale @ torch.stack((PX, PY)).view(2,-1)).reshape((2, *PX.shape))


@lru_cache(maxsize=32)
def quad_table(n, p, dtype, device):
    """
    from: https://pomax.github.io/bezierinfo/legendre-gauss.html
    """
    abscissa, weights = roots_legendre(n)

    w = torch.tensor(weights, dtype = dtype, device = device)
    a = torch.tensor(abscissa, dtype = dtype, device = device)
    X, Y = torch.meshgrid(a, a, indexing = "xy")

    W = torch.outer(w,w) / 4.

    X, Y = p @ (torch.stack((X, Y)).view(2,-1) / 2.)
    
    return X, Y, W.reshape(-1) 

def single_quad_integrate(X, Y, image_header, eval_brightness, eval_parameters, dtype, device, quad_level = 3):
    
    # collect gaussian quadrature weights
    abscissaX, abscissaY, weight = quad_table(quad_level, image_header.pixelscale, dtype, device)
    
    # Specify coordinates at which to evaluate function
    Xs = torch.repeat_interleave(X[...,None], quad_level**2, -1) + abscissaX
    Ys = torch.repeat_interleave(Y[...,None], quad_level**2, -1) + abscissaY

    # Evaluate the model at the quadrature points
    res = eval_brightness(
        X=Xs, Y=Ys, image=image_header, parameters=eval_parameters,
    )

    ref = res.mean(axis = -1)
    # Apply the weights and reduce to original pixel space
    res = (res*weight).sum(axis=-1)
    
    return res, ref
        
def grid_integrate(X, Y, value, compare, image_header, eval_brightness, eval_parameters, dtype, device, tolerance = 1e-2, quad_level = 3, gridding = 5, grid_level = 0, max_level = 2, reference = None):
    if grid_level >= max_level:
        return value

    # Evaluate gaussian quadrature on the specified pixels
    res, ref = single_quad_integrate(X, Y, image_header, eval_brightness, eval_parameters, dtype, device, quad_level = quad_level)

    # Determine which pixels are now converged to sufficient degree
    error = torch.abs((res - ref))
    select = error > (tolerance*reference)

    # Update converged pixels with new value
    value[torch.logical_not(select)] = res[torch.logical_not(select)]

    # Set up sub-gridding to super resolve problem pixels
    stepx, stepy = displacement_grid(gridding,gridding,image_header.pixelscale, dtype, device)
    # Write out the coordinates for the super resolved pixels
    Xs = torch.repeat_interleave(X[select][...,None], gridding**2, -1) + stepx.reshape(-1)
    Ys = torch.repeat_interleave(Y[select][...,None], gridding**2, -1) + stepy.reshape(-1)
    # Copy the current pixel values into new shape with super resolved pixels
    deep_res = torch.repeat_interleave(res[select][...,None], gridding**2, -1)

    # Recursively evaluate the pixels at the higher gridding
    deep_res = grid_integrate(Xs, Ys, deep_res/gridding**2, None, image_header.super_resolve(gridding), eval_brightness, eval_parameters, dtype, device, tolerance, quad_level + 1, gridding, grid_level + 1, max_level, reference = reference*gridding**2)

    # Update the pixels that have been sub-integrated
    value[select] = deep_res.sum(axis=(-1,))
    return value
