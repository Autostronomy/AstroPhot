import torch

from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import gaussian_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], 10 ** F[0]


class GaussianMixin:
    """Gaussian radial light profile.

    The Gaussian profile is a simple and widely used model for extended objects.
    The functional form of the Gaussian profile is defined as:

    $$I(R) = \\frac{{\\rm flux}}{\\sqrt{2\\pi}\\sigma} \\exp(-R^2 / (2 \\sigma^2))$$

    where `I_0` is the intensity at the center of the profile and `sigma` is the
    standard deviation which controls the width of the profile.

    **Parameters:**
    -    `sigma`: Standard deviation of the Gaussian profile in arcseconds.
    -    `flux`: Total flux of the Gaussian profile.
    """

    _model_type = "gaussian"
    _parameter_specs = {
        "sigma": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "flux": {"units": "flux", "valid": (0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            gaussian_np,
            ("sigma", "flux"),
            _x0_func,
        )

    @forward
    def radial_model(self, R: ArrayLike, sigma: ArrayLike, flux: ArrayLike) -> ArrayLike:
        return func.gaussian(R, sigma, flux)


class iGaussianMixin:
    """Gaussian radial light profile.

    The Gaussian profile is a simple and widely used model for extended objects.
    The functional form of the Gaussian profile is defined as:

    $$I(R) = \\frac{{\\rm flux}}{\\sqrt{2\\pi}\\sigma} \\exp(-R^2 / (2 \\sigma^2))$$

    where `sigma` is the standard deviation which controls the width of the
    profile and `flux` gives the total flux of the profile (assuming no
    perturbations).

    `sigma` and `flux` are batched by their first dimension, allowing for
    multiple Gaussian profiles to be defined at once.

    **Parameters:**
    -    `sigma`: Standard deviation of the Gaussian profile in arcseconds.
    -    `flux`: Total flux of the Gaussian profile.
    """

    _model_type = "gaussian"
    _parameter_specs = {
        "sigma": {"units": "arcsec", "valid": (0, None)},
        "flux": {"units": "flux", "valid": (0, None)},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=gaussian_np,
            params=("sigma", "flux"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i: int, R: ArrayLike, sigma: ArrayLike, flux: ArrayLike) -> ArrayLike:
        return func.gaussian(R, sigma[i], flux[i])
