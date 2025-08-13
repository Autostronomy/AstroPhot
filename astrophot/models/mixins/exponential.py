import torch

from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import exponential_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], 10 ** F[4]


class ExponentialMixin:
    """Exponential radial light profile.

    An exponential is a classical radial model used in many contexts. The
    functional form of the exponential profile is defined as:

    $$I(R) = I_e * \\exp\\left(- b_1\\left(\\frac{R}{R_e} - 1\\right)\\right)$$

    Ie is the brightness at the effective radius, and Re is the effective
    radius. $b_1$ is a constant that ensures $I_e$ is the brightness at $R_e$.

    **Parameters:**
    -    `Re`: effective radius in arcseconds
    -    `Ie`: effective surface density in flux/arcsec^2
    """

    _model_type = "exponential"
    _parameter_specs = {
        "Re": {"units": "arcsec", "valid": (0, None), "shape": ()},
        "Ie": {"units": "flux/arcsec^2", "valid": (0, None), "shape": ()},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            exponential_np,
            ("Re", "Ie"),
            _x0_func,
        )

    @forward
    def radial_model(self, R: ArrayLike, Re: ArrayLike, Ie: ArrayLike) -> ArrayLike:
        return func.exponential(R, Re, Ie)


class iExponentialMixin:
    """Exponential radial light profile.

    An exponential is a classical radial model used in many contexts. The
    functional form of the exponential profile is defined as:

    $$I(R) = I_e * \\exp\\left(- b_1\\left(\\frac{R}{R_e} - 1\\right)\\right)$$

    $I_e$ is the brightness at the effective radius, and $R_e$ is the effective
    radius. $b_1$ is a constant that ensures $I_e$ is the brightness at $R_e$.

    `Re` and `Ie` are batched by their first dimension, allowing for multiple
    exponential profiles to be defined at once.

    **Parameters:**
    -    `Re`: effective radius in arcseconds
    -    `Ie`: effective surface density in flux/arcsec^2
    """

    _model_type = "exponential"
    _parameter_specs = {
        "Re": {"units": "arcsec", "valid": (0, None)},
        "Ie": {"units": "flux/arcsec^2", "valid": (0, None)},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=exponential_np,
            params=("Re", "Ie"),
            x0_func=_x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(self, i: int, R: ArrayLike, Re: ArrayLike, Ie: ArrayLike) -> ArrayLike:
        return func.exponential(R, Re[i], Ie[i])
