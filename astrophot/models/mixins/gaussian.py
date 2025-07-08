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
        "flux": {"units": "flux", "shape": ()},
    }
    _overload_parameter_specs = {
        "logflux": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "flux",
            "overload_function": lambda p: 10**p.logflux.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logflux"):
            return

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: gaussian_np(r, x[0], 10 ** x[1]),
            ("sigma", "logflux"),
            _x0_func,
        )

    @forward
    def radial_model(self, R, sigma, flux):
        return func.gaussian(R, sigma, flux)


class iGaussianMixin:

    _model_type = "gaussian"
    _parameter_specs = {
        "sigma": {"units": "arcsec", "valid": (0, None)},
        "flux": {"units": "flux"},
    }
    _overload_parameter_specs = {
        "logflux": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "flux",
            "overload_function": lambda p: 10**p.logflux.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logflux"):
            return

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=lambda r, *x: gaussian_np(r, x[0], 10 ** x[1]),
            params=("sigma", "logflux"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, sigma, flux):
        return func.gaussian(R, sigma[i], flux[i])
