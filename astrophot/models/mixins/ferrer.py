import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from ...utils.parametric_profiles import ferrer_np
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[5], 1, 1, 10 ** F[0]


class FerrerMixin:

    _model_type = "ferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, 10), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "valid": (0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            ferrer_np,
            ("rout", "alpha", "beta", "I0"),
            x0_func,
        )

    @forward
    def radial_model(self, R, rout, alpha, beta, I0):
        return func.ferrer(R, rout, alpha, beta, I0)


class iFerrerMixin:

    _model_type = "ferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, 10), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "valid": (0.0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=ferrer_np,
            params=("rout", "alpha", "beta", "I0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, rout, alpha, beta, I0):
        return func.ferrer(R, rout[i], alpha[i], beta[i], I0[i])
