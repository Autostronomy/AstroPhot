import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import exponential_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], F[4]


class ExponentialMixin:
    """Mixin for models that use an exponential profile for the radial light
    profile. The functional form of the exponential profile is defined as:

    I(R) = Ie * exp(- (R / Re))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness at the
    effective radius, and Re is the effective radius.

    Parameters:
        Re: effective radius in arcseconds
        Ie: effective surface density in flux/arcsec^2
    """

    _model_type = "exponential"
    _parameter_specs = {
        "Re": {"units": "arcsec", "valid": (0, None)},
        "Ie": {"units": "flux/arcsec^2"},
    }
    _overload_parameter_specs = {
        "logIe": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "Ie",
            "overload_function": lambda p: 10**p.logIe.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logIe"):
            return

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: exponential_np(r, x[0], 10 ** x[1]),
            ("Re", "logIe"),
            _x0_func,
        )

    @forward
    def radial_model(self, R, Re, Ie):
        return func.exponential(R, Re, Ie)


class iExponentialMixin:
    """Mixin for models that use an exponential profile for the radial light
    profile. The functional form of the exponential profile is defined as:

    I(R) = Ie * exp(- (R / Re))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness at the
    effective radius, and Re is the effective radius.

    Parameters:
        Re: effective radius in arcseconds
        Ie: effective surface density in flux/arcsec^2
    """

    _model_type = "exponential"
    _parameter_specs = {
        "Re": {"units": "arcsec", "valid": (0, None)},
        "Ie": {"units": "flux/arcsec^2"},
    }
    _overload_parameter_specs = {
        "logIe": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "Ie",
            "overload_function": lambda p: 10**p.logIe.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logIe"):
            return

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=lambda r, *x: exponential_np(r, x[0], 10 ** x[1]),
            params=("Re", "logIe"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, Re, Ie):
        return func.exponential(R, Re[i], Ie[i])
