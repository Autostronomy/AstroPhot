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
        "n": {"units": "none", "limits": (0.1, 10), "uncertainty": 0.05},
        "Rd": {"units": "arcsec", "limits": (0, None)},
        "I0": {"units": "flux/arcsec^2"},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self, **kwargs):
        super().initialize()

        parametric_initialize(
            self, self.target[self.window], moffat_np, ("n", "Re", "Ie"), _x0_func
        )

    @forward
    def radial_model(self, R, n, Rd, I0):
        return func.moffat(R, n, Rd, I0)


class iMoffatMixin:

    _model_type = "moffat"
    _parameter_specs = {
        "n": {"units": "none", "limits": (0.1, 10), "uncertainty": 0.05},
        "Rd": {"units": "arcsec", "limits": (0, None)},
        "I0": {"units": "flux/arcsec^2"},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self, **kwargs):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=moffat_np,
            params=("n", "Rd", "I0"),
            x0_func=_x0_func,
            segments=self.rays,
        )

    @forward
    def radial_model(self, i, R, n, Rd, I0):
        return func.moffat(R, n[i], Rd[i], I0[i])
