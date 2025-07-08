import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import moffat_np
from .. import func


def _x0_func(model_params, R, F):
    return 2.0, R[4], F[0]


class MoffatMixin:

    _model_type = "moffat"
    _parameter_specs = {
        "n": {"units": "none", "valid": (0.1, 10), "shape": ()},
        "Rd": {"units": "arcsec", "valid": (0, None), "shape": ()},
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
            lambda r, *x: moffat_np(r, x[0], x[1], 10 ** x[2]),
            ("n", "Rd", "logI0"),
            _x0_func,
        )

    @forward
    def radial_model(self, R, n, Rd, I0):
        return func.moffat(R, n, Rd, I0)


class iMoffatMixin:

    _model_type = "moffat"
    _parameter_specs = {
        "n": {"units": "none", "valid": (0.1, 10)},
        "Rd": {"units": "arcsec", "valid": (0, None)},
        "I0": {"units": "flux/arcsec^2"},
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
            prof_func=lambda r, *x: moffat_np(r, x[0], x[1], 10 ** x[2]),
            params=("n", "Rd", "logI0"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, n, Rd, I0):
        return func.moffat(R, n[i], Rd[i], I0[i])
