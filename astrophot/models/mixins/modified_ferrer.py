import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[5], 1, 1, 10 ** F[0]


class ModifiedFerrerMixin:

    _model_type = "modifiedferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, None), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            func.modified_ferrer,
            ("rout", "alpha", "beta", "I0"),
            x0_func,
        )

    @forward
    def radial_model(self, R, rout, alpha, beta, I0):
        return func.modified_ferrer(R, rout, alpha, beta, I0)


class iModifiedFerrerMixin:

    _model_type = "modifiedferrer"
    _parameter_specs = {
        "rout": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, None), "shape": ()},
        "beta": {"units": "unitless", "valid": (0, 2), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=func.modified_ferrer,
            params=("rout", "alpha", "beta", "I0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, rout, alpha, beta, I0):
        return func.modified_ferrer(R, rout[i], alpha[i], beta[i], I0[i])
