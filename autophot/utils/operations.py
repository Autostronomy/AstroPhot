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
    PX, PY = torch.meshgrid(px, py, indexing="xy")
    return (pixelscale @ torch.stack((PX, PY)).view(2, -1)).reshape((2, *PX.shape))


@lru_cache(maxsize=32)
def quad_table(n, p, dtype, device):
    """
    from: https://pomax.github.io/bezierinfo/legendre-gauss.html
    """
    abscissa, weights = roots_legendre(n)

    w = torch.tensor(weights, dtype=dtype, device=device)
    a = torch.tensor(abscissa, dtype=dtype, device=device)
    X, Y = torch.meshgrid(a, a, indexing="xy")

    W = torch.outer(w, w) / 4.0

    X, Y = p @ (torch.stack((X, Y)).view(2, -1) / 2.0)

    return X, Y, W.reshape(-1)


def single_quad_integrate(
    X, Y, image_header, eval_brightness, eval_parameters, dtype, device, quad_level=3
):

    # collect gaussian quadrature weights
    abscissaX, abscissaY, weight = quad_table(
        quad_level, image_header.pixelscale, dtype, device
    )
    # Specify coordinates at which to evaluate function
    Xs = torch.repeat_interleave(X[..., None], quad_level ** 2, -1) + abscissaX
    Ys = torch.repeat_interleave(Y[..., None], quad_level ** 2, -1) + abscissaY

    # Evaluate the model at the quadrature points
    res = eval_brightness(
        X=Xs,
        Y=Ys,
        image=image_header,
        parameters=eval_parameters,
    )

    # Reference flux for pixel is simply the mean of the evaluations
    ref = res[..., (quad_level**2) // 2] #res.mean(axis=-1) # # alternative, use midpoint
    
    # Apply the weights and reduce to original pixel space
    res = (res * weight).sum(axis=-1)

    return res, ref

def grid_integrate(
    X,
    Y,
    image_header,
    eval_brightness,
    eval_parameters,
    dtype,
    device,
    quad_level=3,
    gridding=5,
    _current_depth=1,
    max_depth=2,
    reference=None,
):
    """The grid_integrate function performs adaptive quadrature
    integration over a given pixel grid, offering precision control
    where it is needed most.

    Args:
      X (torch.Tensor): A 2D tensor representing the x-coordinates of the grid on which the function will be integrated.
      Y (torch.Tensor): A 2D tensor representing the y-coordinates of the grid on which the function will be integrated.
      image_header (ImageHeader): An object containing meta-information about the image.
      eval_brightness (callable): A function that evaluates the brightness at each grid point. This function should be compatible with PyTorch tensor operations.
      eval_parameters (Parameter_Group): An object containing parameters that are passed to the eval_brightness function.
      dtype (torch.dtype): The data type of the output tensor. The dtype argument should be a valid PyTorch data type.
      device (torch.device): The device on which to perform the computations. The device argument should be a valid PyTorch device.
      quad_level (int, optional): The initial level of quadrature used in the integration. Defaults to 3.
      gridding (int, optional): The factor by which the grid is subdivided when the integration error for a pixel is above the allowed threshold. Defaults to 5.
      _current_depth (int, optional): The current depth level of the grid subdivision. Used for recursive calls to the function. Defaults to 1.
      max_depth (int, optional): The maximum depth level of grid subdivision. Once this level is reached, no further subdivision is performed. Defaults to 2.
      reference (torch.Tensor or None, optional): A scalar value that represents the allowed threshold for the integration error. 

    Returns:
      torch.Tensor: A tensor of the same shape as X and Y that represents the result of the integration on the grid.

    This function operates by first performing a quadrature
    integration over the given pixels. If the maximum depth level has
    been reached, it simply returns the result. Otherwise, it
    calculates the integration error for each pixel and selects those
    that have an error above the allowed threshold. For pixels that
    have low error, the result is set as computed. For those with high
    error, it sets up a finer sampling grid and recursively evaluates
    the quadrature integration on it. Finally, it integrates the
    results from the finer sampling grid back to the current
    resolution.

    """

    # perform quadrature integration on the given pixels
    res, ref = single_quad_integrate(
        X,
        Y,
        image_header,
        eval_brightness,
        eval_parameters,
        dtype,
        device,
        quad_level=quad_level,
    )

    # if the max depth is reached, simply return the integrated pixels
    if _current_depth >= max_depth:
        return res

    # Begin integral
    integral = torch.zeros_like(X)

    # Select pixels which have errors above the allowed threshold
    select = torch.abs((res - ref)) > reference

    # For pixels with low error, set the results as computed
    integral[torch.logical_not(select)] = res[torch.logical_not(select)]

    # Set up sub-gridding to super resolve problem pixels
    stepx, stepy = displacement_grid(
        gridding, gridding, image_header.pixelscale, dtype, device
    )
    # Write out the coordinates for the super resolved pixels
    subgridX = torch.repeat_interleave(
        X[select].unsqueeze(-1), gridding ** 2, -1
    ) + stepx.reshape(-1)
    subgridY = torch.repeat_interleave(
        Y[select].unsqueeze(-1), gridding ** 2, -1
    ) + stepy.reshape(-1)

    # Recursively evaluate the quadrature integration on the finer sampling grid
    subgridres = grid_integrate(
        subgridX,
        subgridY,
        image_header.super_resolve(gridding),
        eval_brightness,
        eval_parameters,
        dtype,
        device,
        quad_level=quad_level+2,
        gridding=gridding,
        _current_depth=_current_depth+1,
        max_depth=max_depth,
        reference=reference * gridding**2,        
    )

    # Integrate the finer sampling grid back to current resolution
    integral[select] = subgridres.sum(axis=(-1,))

    return integral
    
