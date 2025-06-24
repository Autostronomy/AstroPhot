import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import nuker_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], 10 ** F[4], 1.0, 2.0, 0.5


class NukerMixin:

    _model_type = "nuker"
    _parameter_specs = {
        "Rb": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "Ib": {"units": "flux/arcsec^2", "shape": ()},
        "alpha": {"units": "none", "valid": (0, None), "shape": ()},
        "beta": {"units": "none", "valid": (0, None), "shape": ()},
        "gamma": {"units": "none", "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            nuker_np,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    @forward
    def radial_model(self, R, Rb, Ib, alpha, beta, gamma):
        return func.nuker(R, Rb, Ib, alpha, beta, gamma)


class iNukerMixin:

    _model_type = "nuker"
    _parameter_specs = {
        "Rb": {"units": "arcsec", "valid": (0, None)},
        "Ib": {"units": "flux/arcsec^2"},
        "alpha": {"units": "none", "valid": (0, None)},
        "beta": {"units": "none", "valid": (0, None)},
        "gamma": {"units": "none"},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=nuker_np,
            params=("Rb", "Ib", "alpha", "beta", "gamma"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, Rb, Ib, alpha, beta, gamma):
        return func.nuker(R, Rb[i], Ib[i], alpha[i], beta[i], gamma[i])
