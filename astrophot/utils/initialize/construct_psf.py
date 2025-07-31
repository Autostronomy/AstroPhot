import numpy as np


def gaussian_psf(sigma, img_width, pixelscale, upsample=4, normalize=True):
    """
    create a gaussian point spread function (PSF) image.

    **Args:**
    -  `sigma`: Standard deviation of the Gaussian in arcseconds.
    -  `img_width`: Width of the PSF image in pixels.
    -  `pixelscale`: Pixel scale in arcseconds per pixel.
    -  `upsample`: Upsampling factor to more accurately create the PSF (the outputted PSF is not upsampled).
    -  `normalize`: Whether to normalize the PSF so that the sum of all pixels equals 1. If False, the PSF will not be normalized.
    """
    assert img_width % 2 == 1, "psf images should have an odd shape"

    # Number of super sampled pixels
    N = img_width * upsample
    # Grid of centered pixel locations
    XX, YY = np.meshgrid(
        np.linspace(
            -(N - 1) * pixelscale / (2 * upsample),
            (N - 1) * pixelscale / (2 * upsample),
            N,
        ),
        np.linspace(
            -(N - 1) * pixelscale / (2 * upsample),
            (N - 1) * pixelscale / (2 * upsample),
            N,
        ),
    )
    # Evaluate the Gaussian at each pixel
    ZZ = np.exp(-0.5 * (XX**2 + YY**2) / sigma**2)

    # Reduce the super-sampling back to native resolution
    ZZ = ZZ.reshape(img_width, upsample, img_width, upsample).sum(axis=(1, 3)) / (upsample**2)

    # Normalize the PSF
    if normalize:
        return ZZ / np.sum(ZZ)
    return ZZ


def moffat_psf(n, Rd, img_width, pixelscale, upsample=4, normalize=True):
    """
    Create a Moffat point spread function (PSF) image.

    **Args:**
    -  `n`: Moffat index (power-law index).
    -  `Rd`: Scale radius of the Moffat profile in arcseconds.
    -  `img_width`: Width of the PSF image in pixels.
    -  `pixelscale`: Pixel scale in arcseconds per pixel.
    -  `upsample`: Upsampling factor to more accurately create the PSF (the outputted PSF is not upsampled).
    -  `normalize`: Whether to normalize the PSF so that the sum of all pixels equals 1. If False, the PSF will not be normalized.
    """
    assert img_width % 2 == 1, "psf images should have an odd shape"

    # Number of super sampled pixels
    N = img_width * upsample
    # Grid of centered pixel locations
    XX, YY = np.meshgrid(
        np.linspace(
            -(N - 1) * pixelscale / (2 * upsample),
            (N - 1) * pixelscale / (2 * upsample),
            N,
        ),
        np.linspace(
            -(N - 1) * pixelscale / (2 * upsample),
            (N - 1) * pixelscale / (2 * upsample),
            N,
        ),
    )
    # Evaluate the Moffat at each pixel
    ZZ = 1.0 / (1.0 + (XX**2 + YY**2) / (Rd**2)) ** n

    # Reduce the super-sampling back to native resolution
    ZZ = ZZ.reshape(img_width, upsample, img_width, upsample).sum(axis=(1, 3)) / (upsample**2)

    # Normalize the PSF
    if normalize:
        return ZZ / np.sum(ZZ)
    return ZZ
