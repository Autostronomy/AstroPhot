import torch


def lanczos_1d(x, order):
    """1D Lanczos kernel with window size `order`."""
    mask = (x.abs() < order).to(x.dtype)
    return torch.sinc(x) * torch.sinc(x / order) * mask


def lanczos_kernel(dx, dy, order):
    grid = torch.arange(-order, order + 1, dtype=dx.dtype, device=dx.device)
    lx = lanczos_1d(grid - dx, order)
    ly = lanczos_1d(grid - dy, order)
    kernel = torch.outer(ly, lx)
    return kernel / kernel.sum()


def bilinear_kernel(dx, dy):
    """Bilinear kernel for sub-pixel shifting."""
    kernel = torch.tensor(
        [
            [1 - dx, dx],
            [dy, 1 - dy],
        ],
        dtype=dx.dtype,
        device=dx.device,
    )
    return kernel


def convolve_and_shift(image, shift_kernel, psf):

    image_fft = torch.fft.rfft2(image, s=image.shape)
    psf_fft = torch.fft.rfft2(psf, s=image.shape)
    shift_fft = torch.fft.rfft2(shift_kernel, s=image.shape)

    convolved_fft = image_fft * psf_fft * shift_fft
    return torch.fft.irfft2(convolved_fft, s=image.shape)
