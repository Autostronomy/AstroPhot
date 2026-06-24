from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from ...utils.parametric_profiles import nuker_np
from .. import func


def _x0_func(model_params, R, F):
    return R[4], 10 ** F[4], 1.0, 2.0, 0.5


class NukerMixin:
    """Nuker radial light profile (Lauer et al. 1995).

    This is a classic profile used widely in galaxy modelling. The functional
    form of the Nuker profile is defined as:

    .. math::

       I(R) = I_b2^{\\frac{\\beta - \\gamma}{\\alpha}}\\left(\\frac{R}{R_b}\\right)^{-\\gamma}\\left[1 + \\left(\\frac{R}{R_b}\\right)^{\\alpha}\\right]^{\\frac{\\gamma-\\beta}{\\alpha}}

    It is effectively a double power law profile. :math:`\\gamma` gives the inner
    slope, :math:`\\beta` gives the outer slope, :math:`\\alpha` is somewhat degenerate with
    the other slopes.

    :param Rb: scale length radius
    :param Ib: intensity at the scale length
    :param alpha: sharpness of transition between power law slopes
    :param beta: outer power law slope
    :param gamma: inner power law slope
    """

    _model_type = "nuker"
    _parameter_specs = {
        "Rb": {
            "units": "arcsec",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "scale length radius",
        },
        "Ib": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "intensity at the scale length",
        },
        "alpha": {
            "units": "none",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "sharpness of transition between power law slopes",
        },
        "beta": {
            "units": "none",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "outer power law slope",
        },
        "gamma": {
            "units": "none",
            "shape": (),
            "dynamic": True,
            "description": "inner power law slope",
        },
    }

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
    def radial_model(
        self,
        R: ArrayLike,
        Rb: ArrayLike,
        Ib: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        gamma: ArrayLike,
    ) -> ArrayLike:
        return func.nuker(R, Rb, Ib, alpha, beta, gamma)


class iNukerMixin:
    """Nuker radial light profile (Lauer et al. 1995).

    This is a classic profile used widely in galaxy modelling. The functional
    form of the Nuker profile is defined as:

    .. math::

       I(R) = I_b2^{\\frac{\\beta - \\gamma}{\\alpha}}\\left(\\frac{R}{R_b}\\right)^{-\\gamma}\\left[1 + \\left(\\frac{R}{R_b}\\right)^{\\alpha}\\right]^{\\frac{\\gamma-\\beta}{\\alpha}}

    It is effectively a double power law profile. :math:`\\gamma` gives the inner
    slope, :math:`\\beta` gives the outer slope, :math:`\\alpha` is somewhat degenerate with
    the other slopes.

    ``Rb``, ``Ib``, ``alpha``, ``beta``, and ``gamma`` are batched by their first
    dimension, allowing for multiple Nuker profiles to be defined at once.

    :param Rb: scale length radius
    :param Ib: intensity at the scale length
    :param alpha: sharpness of transition between power law slopes
    :param beta: outer power law slope
    :param gamma: inner power law slope
    """

    _model_type = "nuker"
    _parameter_specs = {
        "Rb": {
            "units": "arcsec",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "scale length radius",
        },
        "Ib": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "intensity at the scale length",
        },
        "alpha": {
            "units": "none",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "sharpness of transition between power law slopes",
        },
        "beta": {
            "units": "none",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "outer power law slope",
        },
        "gamma": {
            "units": "none",
            "shape": (None,),
            "dynamic": True,
            "description": "inner power law slope",
        },
    }

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
    def iradial_model(
        self,
        i: int,
        R: ArrayLike,
        Rb: ArrayLike,
        Ib: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        gamma: ArrayLike,
    ) -> ArrayLike:
        return func.nuker(R, Rb[i], Ib[i], alpha[i], beta[i], gamma[i])


class NukerPSFMixin:
    """Nuker radial light profile (Lauer et al. 1995).

    This is a classic profile used widely in galaxy modelling. The functional
    form of the Nuker profile is defined as:

    .. math::

       I(R) = I_b2^{\\frac{\\beta - \\gamma}{\\alpha}}\\left(\\frac{R}{R_b}\\right)^{-\\gamma}\\left[1 + \\left(\\frac{R}{R_b}\\right)^{\\alpha}\\right]^{\\frac{\\gamma-\\beta}{\\alpha}}

    It is effectively a double power law profile. :math:`\\gamma` gives the inner
    slope, :math:`\\beta` gives the outer slope, :math:`\\alpha` is somewhat degenerate with
    the other slopes.

    :param Rb: scale length radius [pix]
    :param Ib: intensity at the scale length [flux/pix^2]
    :param alpha: sharpness of transition between power law slopes
    :param beta: outer power law slope
    :param gamma: inner power law slope
    """

    _model_type = "nuker"
    _parameter_specs = {
        "Rb": {
            "units": "pix",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "scale length radius [pix]",
        },
        "Ib": {
            "units": "flux/pix^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": False,
            "value": 1.0,
            "description": "intensity at the scale length [flux/pix^2]",
        },
        "alpha": {
            "units": "none",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "sharpness of transition between power law slopes",
        },
        "beta": {
            "units": "none",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "outer power law slope",
        },
        "gamma": {
            "units": "none",
            "shape": (),
            "dynamic": True,
            "description": "inner power law slope",
        },
    }

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
    def radial_model(
        self,
        R: ArrayLike,
        Rb: ArrayLike,
        Ib: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        gamma: ArrayLike,
    ) -> ArrayLike:
        return func.nuker(R, Rb, Ib, alpha, beta, gamma)
