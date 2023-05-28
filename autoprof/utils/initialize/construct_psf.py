import numpy as np

from .center import Lanczos_peak, center_of_mass, GaussianDensity_Peak
from ..interpolate import shift_Lanczos_np, point_Lanczos


def gaussian_psf(sigma, img_width, pixelscale, upsample=4):
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
    ZZ = np.exp(-0.5 * (XX ** 2 + YY ** 2) / sigma ** 2)

    # Reduce the super-sampling back to native resolution
    ZZ = ZZ.reshape(img_width, upsample, img_width, upsample).sum(axis=(1, 3)) / (
        upsample ** 2
    )

    # Normalize the PSF
    return ZZ / np.sum(ZZ)


def moffat_psf(n, Rd, img_width, pixelscale, upsample=4):
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
    ZZ = 1.0 / (1.0 + (XX ** 2 + YY ** 2) / (Rd ** 2)) ** n

    # Reduce the super-sampling back to native resolution
    ZZ = ZZ.reshape(img_width, upsample, img_width, upsample).sum(axis=(1, 3)) / (
        upsample ** 2
    )

    # Normalize the PSF
    return ZZ / np.sum(ZZ)


def construct_psf(
    stars, image, sky_est, size=51, mask=None, keep_init=False, Lanczos_scale=3
):
    """Given a list of initial guesses for star center locations, finds
    the interpolated flux peak, re-centers the stars such that they
    are exactly on a pixel center, the median stacks the normalized
    stars to determine an average PSF.

    Note that all coordinates in this function are pixel
    coordinates. That is, the image[0][0] pixel is at location (0,0)
    and the image[2][7] pixel is at location (2,7) in this coordinate
    system.
    """
    size += 1 - (size % 2)
    star_centers = []
    # determine exact (sub-pixel) center for each star

    for star in stars:
        if keep_init:
            star_centers = list(np.array(s) for s in stars)
            break
        try:
            peak = GaussianDensity_Peak(star, image)
        except Exception as e:
            AP_config.ap_logger.warning("issue finding star center")
            AP_config.ap_logger.warning(e)
            AP_config.ap_logger.warning("skipping")
            continue
        pixel_cen = np.round(peak)
        if (
            pixel_cen[0] < ((size - 1) / 2)
            or pixel_cen[0] > (image.shape[1] - ((size - 1) / 2) - 1)
            or pixel_cen[1] < ((size - 1) / 2)
            or pixel_cen[1] > (image.shape[0] - ((size - 1) / 2) - 1)
        ):
            AP_config.ap_logger.debug("skipping star near edge at: {peak}")
            continue
        star_centers.append(peak)

    stacking = []
    # Extract the star from the image, and shift to align exactly with pixel grid
    for star in star_centers:
        center = np.round(star)
        border = int((size - 1) / 2 + Lanczos_scale)
        I = image[
            int(center[1] - border) : int(center[1] + border + 1),
            int(center[0] - border) : int(center[0] + border + 1),
        ]
        shift = center - star
        I = shift_Lanczos_np(I - sky_est, shift[0], shift[1], scale=Lanczos_scale)
        I = I[Lanczos_scale:-Lanczos_scale, Lanczos_scale:-Lanczos_scale]
        border = (size - 1) / 2
        if mask is not None:
            I[
                mask[
                    int(center[1] - border) : int(center[1] + border + 1),
                    int(center[0] - border) : int(center[0] + border + 1),
                ]
            ] = np.nan
        # Add the normalized star image to the list
        stacking.append(I / np.sum(I))

    # Median stack the pixel images
    stacked_psf = np.nanmedian(stacking, axis=0)
    stacked_psf[stacked_psf < 0] = 0

    return stacked_psf / np.sum(stacked_psf)
