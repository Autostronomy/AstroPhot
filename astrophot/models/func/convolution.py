from functools import lru_cache

import torch


def convolve(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:

    image_fft = torch.fft.rfft2(image, s=image.shape)
    psf_fft = torch.fft.rfft2(psf, s=image.shape)

    convolved_fft = image_fft * psf_fft
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
