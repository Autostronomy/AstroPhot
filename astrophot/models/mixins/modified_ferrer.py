import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from ...utils.parametric_profiles import modified_ferrer_np
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[5], 1, 1, F[0]


class ModifiedFerrerMixin:

    _model_type = "modifiedferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, 10), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }
    _overload_parameter_specs = {
        "logI0": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "I0",
            "overload_function": lambda p: 10**p.logI0.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logI0"):
            return

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: modified_ferrer_np(r, x[0], x[1], x[2], 10 ** x[3]),
            ("rout", "alpha", "beta", "logI0"),
            x0_func,
        )

    @forward
    def radial_model(self, R, rout, alpha, beta, I0):
        return func.modified_ferrer(R, rout, alpha, beta, I0)


class iModifiedFerrerMixin:

    _model_type = "modifiedferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, 10), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }
    _overload_parameter_specs = {
        "logI0": {
            "units": "log10(flux/arcsec^2)",
            "shape": (),
            "overloads": "I0",
            "overload_function": lambda p: 10**p.logI0.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # Only auto initialize for standard parametrization
        if not hasattr(self, "logI0"):
            return

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=lambda r, *x: modified_ferrer_np(r, x[0], x[1], x[2], 10 ** x[3]),
            params=("rout", "alpha", "beta", "logI0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, rout, alpha, beta, I0):
        return func.modified_ferrer(R, rout[i], alpha[i], beta[i], I0[i])
