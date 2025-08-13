from functools import lru_cache

from ...backend_obj import backend, ArrayLike


def convolve(image: ArrayLike, psf: ArrayLike) -> ArrayLike:

    image_fft = backend.fft.rfft2(image, s=image.shape)
    psf_fft = backend.fft.rfft2(psf, s=image.shape)

    convolved_fft = image_fft * psf_fft
    convolved = backend.fft.irfft2(convolved_fft, s=image.shape)
    return backend.roll(
        convolved,
        shifts=(-(psf.shape[0] // 2), -(psf.shape[1] // 2)),
        dims=(0, 1),
    )


@lru_cache(maxsize=32)
def curvature_kernel(dtype, device):
    kernel = backend.as_array(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],  # [[1., -2.0, 1.], [-2.0, 4, -2.0], [1.0, -2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return kernel
