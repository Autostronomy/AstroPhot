import numpy as np
import astrophot as ap
from astropy.wcs import WCS


def get_astropy_wcs():
    hdr = {
        "SIMPLE": "T",
        "NAXIS": 2,
        "NAXIS1": 180,
        "NAXIS2": 170,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 195.0588,
        "CRVAL2": 28.0608,
        "CRPIX1": 90.5,
        "CRPIX2": 85.5,
        "CD1_1": -0.000416666666666667,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.000416666666666667,
        # "IMAGEW": 180.0,
        # "IMAGEH": 170.0,
    }
    return WCS(hdr)


def make_basic_sersic(
    N=52,
    M=50,
    pixelscale=0.8,
    x=20.5,
    y=21.4,
    PA=45 * np.pi / 180,
    q=0.7,
    n=1.5,
    Re=15.1,
    Ie=10.0,
    rand=12345,
    **kwargs,
):

    np.random.seed(rand)
    mask = np.zeros((N, M), dtype=bool)
    mask[0][0] = True
    target = ap.TargetImage(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
        mask=mask,
        zeropoint=21.5,
        **kwargs,
    )

    MODEL = ap.models.SersicGalaxy(
        name="basic sersic model",
        target=target,
        center=[x, y],
        PA=PA,
        q=q,
        n=n,
        Re=Re,
        Ie=Ie,
        sampling_mode="quad:5",
    )

    img = ap.backend.to_numpy(MODEL().data.T)
    target.data = (
        img
        + np.random.normal(scale=0.5, size=img.shape)
        + np.random.normal(scale=np.sqrt(img) / 10)
    )
    target.variance = 0.5**2 + img / 100

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
    target = ap.TargetImage(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
    )

    MODEL = ap.models.GaussianGalaxy(
        name="basic gaussian source",
        target=target,
        center=[x, y],
        sigma=sigma,
        flux=flux,
        PA=0.0,
        q=0.99,
    )

    img = ap.backend.to_numpy(MODEL().data.T)
    target.data = (
        img
        + np.random.normal(scale=0.1, size=img.shape)
        + np.random.normal(scale=np.sqrt(img) / 10)
    )
    target.variance = 0.1**2 + img / 100

    return target


def make_basic_gaussian_psf(
    N=25,
    pixelscale=0.8,
    sigma=4,
    rand=12345,
):

    np.random.seed(rand)
    psf = ap.utils.initialize.gaussian_psf(sigma * pixelscale, N, pixelscale)
    target = ap.PSFImage(
        data=psf + np.random.normal(scale=np.sqrt(psf) / 20),
        pixelscale=pixelscale,
        variance=psf / 400,
    )
    target.normalize()

    return target
