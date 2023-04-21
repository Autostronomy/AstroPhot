from typing import Callable, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, convolve_fft
from scipy.fft import next_fast_len

def fft_convolve_torch(img, psf, psf_fft=False, img_prepadded=False):
    # Ensure everything is tensor
    img = torch.as_tensor(img)
    psf = torch.as_tensor(psf)

    if img_prepadded:
        s = img.size()
    else:
        s = tuple(next_fast_len(int(d+(p+1)/2), real = True) for d,p in zip(img.size(), psf.size())) #list(int(d + (p + 1) / 2) for d, p in zip(img.size(), psf.size()))

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

def displacement_spacing(N, dtype = torch.float64, device = "cpu"):
    return torch.linspace(-(N - 1)/(2*N), (N - 1)/(2*N), N, dtype = dtype, device = device)
    
def displacement_grid(*N, pixelscale = 1., dtype = torch.float64, device = "cpu"):
    return torch.meshgrid(*tuple(displacement_spacing(n, dtype = dtype, device = device)*pixelscale for n in N), indexing = "xy")
    
def selective_integrate(
        X: torch.Tensor,
        Y: torch.Tensor,
        data: torch.Tensor,
        image_header: "Image_Header",
        eval_brightness: Callable,
        max_depth: int = 3,
        _depth: int = 1,
        _reference_brightness: Optional[float] = None,
        integrate_threshold: float = 1e-2,
):
    """Sample the model at higher resolution than the input image.
    
    This function selectively refines the integration of an input
    image based on the local curvature of the image data.  It
    recursively evaluates the model at higher resolutions in areas
    where the curvature exceeds the specified threshold.  With
    each level of recursion, the function refines the affected
    areas using a 3x3 grid for super-resolution.

    Args:
      X (torch.tensor): A tensor representing the X coordinates of the input image.
      Y (torch.tensor): A tensor representing the Y coordinates of the input image.
      data (torch.tensor): A tensor containing the input image data.
      image_header (Image_Header): An instance of the Image_Header class containing the image's header information.
      eval_brightness (Callable): Function which evaluates the brightness at a given coordinate.
      _depth (int, optional): The current recursion depth. Default is 1.
      max_depth (int, optional): The maximum recursion depth allowed. Default is 3.
      _reference_brightness (float or None, optional): The reference brightness value used to normalize the curvature
                                                       values. If None, the maximum value of the input data divided by
                                                       10 will be used. Default is None.

    Returns:
        None. The function updates the input data tensor in-place with the selectively integrated values.
  
    """
    # check recursion depth, exit if too deep
    if _depth > max_depth:
        return
        
    with torch.no_grad():
        if _reference_brightness is None:
            _reference_brightness = torch.max(data)/10
        curvature_kernel = torch.tensor([[0,1.,0],[1.,-4,1.],[0,1.,0]], device = data.device, dtype = data.dtype)
        if _depth == 1:
            curvature = torch.abs(fft_convolve_torch(data, curvature_kernel))
            curvature[:,0] = 0
            curvature[:,-1] = 0
            curvature[0,:] = 0
            curvature[-1,:] = 0
            curvature /= _reference_brightness
            select = curvature > integrate_threshold
        else:
            curvature = torch.sum(data * curvature_kernel, axis = (1,2)) / _reference_brightness
            select = curvature > integrate_threshold
            select = select.view(-1,1,1).repeat(1,3,3)
        
        # compute the subpixel coordinate shifts for even integration within a pixel 
        shiftsx, shiftsy = displacement_grid(3, 3, pixelscale = image_header.pixelscale, device = data.device, dtype = data.dtype)
                        
    # Reshape coordinates to add two dimensions with the super-resolved coordiantes
    Xs = X[select].view(-1,1,1).repeat(1,3,3) + shiftsx
    Ys = Y[select].view(-1,1,1).repeat(1,3,3) + shiftsy
    # evaluate the model on the new smaller coordinate grid in each pixel
    res = eval_brightness(image = image_header.super_resolve(3), X = Xs, Y = Ys)
    
    # Apply recursion to integrate any further pixels as needed
    selective_integrate(
        X = Xs,
        Y = Ys,
        data = res,
        image_header = image_header.super_resolve(3),
        eval_brightness = eval_brightness,
        _depth = _depth+1,
        max_depth = max_depth,
        _reference_brightness = _reference_brightness,
        integrate_threshold = integrate_threshold,
    )
    
    # Update the pixels with the new integrated values
    data[select] = res.sum(axis = (1,2))
