from functools import lru_cache

import torch


def lanczos_1d(x, order):
    """1D Lanczos kernel with window size `order`."""
    mask = (x.abs() < order).to(x.dtype)
    return torch.sinc(x) * torch.sinc(x / order) * mask


def lanczos_kernel(di, dj, order):
    grid = torch.arange(-order, order + 1, dtype=di.dtype, device=di.device)
    li = lanczos_1d(grid - di, order)
    lj = lanczos_1d(grid - dj, order)
    kernel = torch.outer(li, lj)
    return kernel / kernel.sum()


def bilinear_kernel(di, dj):
    """Bilinear kernel for sub-pixel shifting."""
    w00 = (1 - di) * (1 - dj)
    w10 = di * (1 - dj)
    w01 = (1 - di) * dj
    w11 = di * dj

    kernel = torch.stack([w00, w10, w01, w11]).reshape(2, 2)
    return kernel


def fft_shift_kernel(shape, di, dj):
    """FFT shift theorem gives "exact" shift in phase space. Not really exact for DFT"""
    ni, nj = shape
    ki = torch.fft.fftfreq(ni, dtype=di.dtype, device=di.device)
    kj = torch.fft.rfftfreq(nj, dtype=di.dtype, device=di.device)

    Ki, Kj = torch.meshgrid(ki, kj, indexing="ij")
    phase = -2j * torch.pi * (Ki * torch.arctan(di) + Kj * torch.arctan(dj))
    return torch.exp(phase)


def convolve(image, psf):

    image_fft = torch.fft.rfft2(image, s=image.shape)
    psf_fft = torch.fft.rfft2(psf, s=image.shape)

    convolved_fft = image_fft * psf_fft
    convolved = torch.fft.irfft2(convolved_fft, s=image.shape)
    return torch.roll(
        convolved,
        shifts=(-(psf.shape[0] // 2), -(psf.shape[1] // 2)),
        dims=(0, 1),
    )


def convolve_and_shift(image, psf, shift):

    image_fft = torch.fft.rfft2(image, s=image.shape)
    psf_fft = torch.fft.rfft2(psf, s=image.shape)

    if shift is None:
        convolved_fft = image_fft * psf_fft
    else:
        shift_kernel = fft_shift_kernel(image.shape, shift[0], shift[1])
        convolved_fft = image_fft * psf_fft * shift_kernel

    convolved = torch.fft.irfft2(convolved_fft, s=image.shape)
    return torch.roll(
        convolved,
        shifts=(-(psf.shape[0] // 2), -(psf.shape[1] // 2)),
        dims=(0, 1),
    )


@lru_cache(maxsize=32)
def curvature_kernel(dtype, device):
    kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],  # [[1., -2.0, 1.], [-2.0, 4, -2.0], [1.0, -2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return kernel
