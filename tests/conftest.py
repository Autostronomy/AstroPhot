"""Shared fixtures for the AstroPhot tests.

Where to look for the tests of a piece of code (the files follow the layout of
the ``astrophot`` package):

=========================  ====================================================
``astrophot/image``        ``test_image.py`` (Image, TargetImage, PSFImage,
                           JacobianImage), ``test_image_wcs.py`` (coordinates
                           against astropy), ``test_image_list.py``,
                           ``test_image_window.py``, ``test_image_cmos.py``,
                           ``test_image_sip.py``
``astrophot/models``       ``test_models.py`` (behaviour common to all models:
                           sampling, derivatives, saving), then one file per
                           family: ``test_models_component.py`` (every galaxy,
                           sky and point model), ``test_models_psf.py``,
                           ``test_models_group.py``, ``test_models_batch.py``
``astrophot/fit``          ``test_fit.py`` (every fitter on one model),
                           ``test_fit_batch.py`` (BatchLM)
``astrophot/param``        ``test_param.py``
``astrophot/plots``        ``test_plots.py``
``astrophot/utils``        ``test_utils.py``
``astrophot/config.py``    ``test_config.py`` (logging)
``astrophot/cli``          ``test_cli.py``
``docs/source/tutorials``  ``test_notebooks.py`` (slow, skip them with
                           ``pytest -m "not notebooks"``)
=========================  ====================================================

Every test runs in its own empty temporary directory, so tests may write files
by relative name and nothing is left behind in the repository.

The fixtures give each test a fresh object, so a test is free to modify what it
is given. The pixel values behind them are only computed once per session.
"""

from functools import lru_cache

import astrophot as ap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

######################################################################
# Environment
######################################################################


@pytest.fixture(autouse=True)
def no_block_show(monkeypatch):
    """Figures are never shown, ``plt.show`` closes them instead."""

    def close_show(*args, **kwargs):
        plt.close("all")

    monkeypatch.setattr(plt, "show", close_show)
    matplotlib.use("Agg")
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run in an empty directory so files written by a test are cleaned up."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def default_logger():
    """Messages are printed and not written to a file, whatever the test before did."""
    yield
    ap.set_logger(filename=None)


@pytest.fixture(autouse=True)
def seed():
    """Same random numbers every run, for the noise and for the samplers."""
    np.random.seed(12345)
    if ap.backend.backend == "torch":
        ap.backend.module.manual_seed(12345)


######################################################################
# Mock data
######################################################################


@lru_cache(maxsize=None)
def _sersic_pixels(N, M, pixelscale, x, y, PA, q, n, Re, Ie, rand):
    """Noisy image of a Sersic model and its variance, as numpy arrays."""
    target = ap.TargetImage(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
    )
    model = ap.models.SersicGalaxy(
        name="mock_sersic_generator",
        target=target,
        center=[x, y],
        PA=PA,
        q=q,
        n=n,
        Re=Re,
        Ie=Ie,
        sampling_mode="quad:5",
    )
    img = ap.backend.to_numpy(model().data)
    rng = np.random.RandomState(rand)
    data = img + rng.normal(scale=0.5, size=img.shape) + rng.normal(scale=np.sqrt(img) / 10)
    return data, 0.5**2 + img / 100


def make_sersic_target(
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
    """TargetImage of a noisy Sersic galaxy, with variance, a PSF and one masked pixel.

    The image has N rows and M columns, the other arguments are the true
    parameters of the galaxy. Extra keyword arguments go to the TargetImage.
    """
    data, variance = _sersic_pixels(N, M, pixelscale, x, y, PA, q, n, Re, Ie, rand)
    mask = np.zeros((N, M), dtype=bool)
    mask[0][0] = True
    return ap.TargetImage(
        data=data.copy(),
        variance=variance.copy(),
        mask=mask,
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
        zeropoint=21.5,
        **kwargs,
    )


def make_gaussian_target(N=50, M=50, pixelscale=0.8, x=24.5, y=25.4, sigma=10, flux=1, rand=12345):
    """TargetImage of a noisy, nearly round, Gaussian with variance and a PSF."""
    target = ap.TargetImage(
        data=np.zeros((N, M)),
        pixelscale=pixelscale,
        psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
    )
    model = ap.models.GaussianGalaxy(
        name="mock_gaussian_generator",
        target=target,
        center=[x, y],
        sigma=sigma,
        flux=flux,
        PA=0.0,
        q=0.99,
    )
    img = ap.backend.to_numpy(model().data)
    rng = np.random.RandomState(rand)
    target.data = img + rng.normal(scale=0.1, size=img.shape) + rng.normal(scale=np.sqrt(img) / 10)
    target.variance = 0.1**2 + img / 100
    return target


def make_psf_target(N=25, pixelscale=0.8, sigma=4, rand=12345):
    """PSFImage of a noisy Gaussian PSF, normalized, with variance."""
    psf = ap.utils.initialize.gaussian_psf(sigma * pixelscale, N, pixelscale)
    rng = np.random.RandomState(rand)
    target = ap.PSFImage(
        data=psf + rng.normal(scale=np.sqrt(psf) / 20),
        upsample=2,
        variance=psf / 400,
    )
    target.normalize()
    return target


@pytest.fixture
def sersic_target():
    """52 by 50 pixel noisy image of a Sersic galaxy, see `make_sersic_target`."""
    return make_sersic_target()


@pytest.fixture
def sersic_target_factory():
    """`make_sersic_target`, for tests that need several targets or another shape."""
    return make_sersic_target


@pytest.fixture
def gaussian_target_factory():
    """`make_gaussian_target`, 50 by 50 pixel noisy image of a Gaussian."""
    return make_gaussian_target


@pytest.fixture
def psf_target():
    """25 by 25 pixel noisy PSFImage of a Gaussian PSF."""
    return make_psf_target()


######################################################################
# Models
######################################################################


@pytest.fixture
def sersic_model(sersic_target):
    """Initialized Sersic model on `sersic_target` that starts away from the truth.

    This is the model to use when a test only needs "some model", and the one
    the fitters are asked to improve.
    """
    model = ap.Model(
        name="test_sersic",
        model_type="sersic galaxy model",
        center=[20, 20],
        PA=np.pi,
        q=0.7,
        n=2,
        Re=15,
        Ie=10.0,
        target=sersic_target,
    )
    model.initialize()
    return model


@pytest.fixture
def true_sersic_model(sersic_target):
    """Sersic model with the parameters that `sersic_target` was made from."""
    return ap.models.SersicGalaxy(
        name="basic_sersic_model",
        target=sersic_target,
        center=[20.5, 21.4],
        PA=45 * np.pi / 180,
        q=0.7,
        n=1.5,
        Re=15.1,
        Ie=10.0,
        sampling_mode="quad:5",
    )


######################################################################
# Plain images
######################################################################


@pytest.fixture
def image():
    """Blank Image with 10 rows and 15 columns, so `_data` has shape (15, 10)."""
    return ap.Image(data=np.zeros((10, 15)), pixelscale=1.0, zeropoint=1.0, name="image")


@pytest.fixture
def image_pair():
    """Two Images with different shape, pixelscale, zeropoint and pixel values (zeros and ones)."""
    image1 = ap.Image(data=np.zeros((10, 15)), pixelscale=1.0, zeropoint=1.0, name="image1")
    image2 = ap.Image(data=np.ones((15, 10)), pixelscale=0.5, zeropoint=2.0, name="image2")
    return image1, image2


@pytest.fixture
def target_image_pair():
    """Two TargetImages like `image_pair`, with variance and mask."""
    image1 = ap.TargetImage(
        data=np.zeros((10, 15)),
        variance=np.ones((10, 15)),
        mask=np.zeros((10, 15)),
        pixelscale=1.0,
        zeropoint=1.0,
        name="image1",
    )
    image2 = ap.TargetImage(
        data=np.ones((15, 10)),
        variance=np.ones((15, 10)),
        mask=np.zeros((15, 10)),
        pixelscale=0.5,
        zeropoint=2.0,
        name="image2",
    )
    return image1, image2


######################################################################
# Plots
######################################################################


@pytest.fixture
def fig_ax():
    """A matplotlib figure and axis, closed after the test."""
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)
