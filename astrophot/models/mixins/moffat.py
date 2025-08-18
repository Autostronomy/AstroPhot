import torch
from torch import Tensor

from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import moffat_np
from .. import func


def _x0_func(model_params, R, F):
    return 2.0, R[4], 10 ** F[0]


class MoffatMixin:
    """Moffat radial light profile (Moffat 1969).

    The moffat profile gives a good representation of the general structure of
    PSF functions for ground based data. It can also be used to fit extended
    objects. The functional form of the Moffat profile is defined as:

    $$I(R) = \\frac{I_0}{(1 + (R/R_d)^2)^n}$$

    `n` is the concentration index which controls the shape of the profile.

    **Parameters:**
    -    `n`: Concentration index which controls the shape of the brightness profile
    -    `Rd`: Scale length radius
    -    `I0`: Intensity at the center of the profile
    """

    _model_type = "moffat"
    _parameter_specs = {
        "n": {"units": "none", "valid": (0.1, 10), "shape": ()},
        "Rd": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "I0": {"units": "flux/arcsec^2", "valid": (0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            moffat_np,
            ("n", "Rd", "I0"),
            _x0_func,
        )

    @forward
    def radial_model(self, R: ArrayLike, n: ArrayLike, Rd: ArrayLike, I0: ArrayLike) -> ArrayLike:
        return func.moffat(R, n, Rd, I0)


class iMoffatMixin:
    """Moffat radial light profile (Moffat 1969).

    The moffat profile gives a good representation of the general structure of
    PSF functions for ground based data. It can also be used to fit extended
    objects. The functional form of the Moffat profile is defined as:

    $$I(R) = \\frac{I_0}{(1 + (R/R_d)^2)^n}$$

    `n` is the concentration index which controls the shape of the profile.

    `n`, `Rd`, and `I0` are batched by their first dimension, allowing for
    multiple Moffat profiles to be defined at once.

    **Parameters:**
    -    `n`: Concentration index which controls the shape of the brightness profile
    -    `Rd`: Scale length radius
    -    `I0`: Intensity at the center of the profile
    """

    _model_type = "moffat"
    _parameter_specs = {
        "n": {"units": "none", "valid": (0.1, 10)},
        "Rd": {"units": "arcsec", "valid": (0, None)},
        "I0": {"units": "flux/arcsec^2", "valid": (0, None)},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=moffat_np,
            params=("n", "Rd", "I0"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(
        self, i: int, R: ArrayLike, n: ArrayLike, Rd: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.moffat(R, n[i], Rd[i], I0[i])
