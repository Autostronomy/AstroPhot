import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[2], R[5], 2, 10 ** F[0]


class EmpiricalKingMixin:

    _model_type = "empiricalking"
    _parameter_specs = {
        "Rc": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "Rt": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, None), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            func.empirical_king,
            ("Rc", "Rt", "alpha", "I0"),
            x0_func,
        )

    @forward
    def radial_model(self, R, Rc, Rt, alpha, I0):
        return func.empirical_king(R, Rc, Rt, alpha, I0)


class iEmpiricalKingMixin:

    _model_type = "empiricalking"
    _parameter_specs = {
        "Rc": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "Rt": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, None), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=func.empirical_king,
            params=("Rc", "Rt", "alpha", "I0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, Rc, Rt, alpha, I0):
        return func.empirical_king(R, Rc[i], Rt[i], alpha[i], I0[i])
