import torch
import numpy as np

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from ...utils.parametric_profiles import king_np
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[2], R[5], 2, 10 ** F[0]


class KingMixin:

    _model_type = "king"
    _parameter_specs = {
        "Rc": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "Rt": {"units": "arcsec", "valid": (0.0, None), "shape": ()},
        "alpha": {"units": "unitless", "valid": (0, None), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "valid": (0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.alpha.initialized:
            self.alpha.dynamic_value = 2.0

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: king_np(r, x[0], x[1], 2.0, x[2]),
            ("Rc", "Rt", "I0"),
            x0_func,
        )

    @forward
    def radial_model(self, R, Rc, Rt, alpha, I0):
        return func.king(R, Rc, Rt, alpha, I0)


class iKingMixin:

    _model_type = "king"
    _parameter_specs = {
        "Rc": {"units": "arcsec", "valid": (0.0, None)},
        "Rt": {"units": "arcsec", "valid": (0.0, None)},
        "alpha": {"units": "unitless", "valid": (0, 10)},
        "I0": {"units": "flux/arcsec^2", "valid": (0, None)},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.alpha.initialized:
            self.alpha.dynamic_value = 2.0 * np.ones(self.segments)
        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=lambda r, *x: king_np(r, x[0], x[1], 2.0, x[2]),
            params=("Rc", "Rt", "I0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i, R, Rc, Rt, alpha, I0):
        return func.king(R, Rc[i], Rt[i], alpha[i], I0[i])
