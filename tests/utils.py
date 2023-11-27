import torch
import numpy as np
import astrophot as ap
from astropy.wcs import WCS

def get_astropy_wcs():
    hdr = {
        "SIMPLE": "T",
        "NAXIS": 2,
        "NAXIS1": 180,
        "NAXIS2": 180,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 195.0588,
        "CRVAL2": 28.0608,
        "CRPIX1": 90.5,
        "CRPIX2": 90.5,
        "CD1_1": -0.000416666666666667,
        "CD1_2": 0.,
        "CD2_1": 0.,
        "CD2_2": 0.000416666666666667,
        "IMAGEW": 180.,
        "IMAGEH": 180.,
    }
    return WCS(hdr)


def make_basic_sersic(
    N=50,
    M=50,
    pixelscale=0.8,
    x=24.5,
    y=25.4,
    PA=45 * np.pi / 180,
    q=0.6,
    n=2,
    Re=7.1,
    Ie=0,
    rand=12345,
):

    np.random.seed(rand)
    mask = np.zeros((N, M), dtype = bool)
    mask[0][0] = True
    target = ap.image.Target_Image(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
        mask = mask,
    )

    MODEL = ap.models.Sersic_Galaxy(
        name="basic sersic model",
        target=target,
        parameters={
            "center": [x, y],
            "PA": PA,
            "q": q,
            "n": n,
            "Re": Re,
            "Ie": Ie,
        },
    )

    img = MODEL().data.detach().cpu().numpy()
    target.data = (
        img
        + np.random.normal(scale=0.1, size=img.shape)
        + np.random.normal(scale=np.sqrt(img) / 10)
    )
    target.variance = 0.1 ** 2 + img / 100

    return target


def make_basic_gaussian(
    N=50,
    M=50,
    pixelscale=0.8,
    x=24.5,
    y=25.4,
    PA=45 * np.pi / 180,
    sigma=3,
    flux=1,
    rand=12345,
):

    np.random.seed(rand)
    target = ap.image.Target_Image(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
    )

    MODEL = ap.models.Gaussian_Galaxy(
        name="basic gaussian source",
        target=target,
        parameters={
            "center": [x, y],
            "sigma": sigma,
            "flux": flux,
            "PA": {"value": 0., "locked": True},
            "q": {"value": 0.99, "locked": True},
        },
    )

    img = MODEL().data.detach().cpu().numpy()
    target.data = (
        img
        + np.random.normal(scale=0.1, size=img.shape)
        + np.random.normal(scale=np.sqrt(img) / 10)
    )
    target.variance = 0.1 ** 2 + img / 100

    return target


def make_basic_gaussian_psf(
    N=25,
    pixelscale=0.8,
    sigma=3,
    rand=12345,
):

    np.random.seed(rand)
    psf = ap.utils.initialize.gaussian_psf(sigma / pixelscale, N, pixelscale)
    psf += np.random.normal(scale = psf / 2)
    psf[psf < 0] = 0
    target = ap.image.PSF_Image(
        data=psf / np.sum(psf),
        pixelscale=pixelscale,
    )

    return target
