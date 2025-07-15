import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import gaussian_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], F[0]


class GaussianMixin:

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
    def radial_model(self, R, sigma, flux):
        return func.gaussian(R, sigma, flux)


class iGaussianMixin:

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
    def iradial_model(self, i, R, sigma, flux):
        return func.gaussian(R, sigma[i], flux[i])
