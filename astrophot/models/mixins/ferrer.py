from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from ...utils.parametric_profiles import ferrer_np
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[5], 1, 1, 10 ** F[0]


class FerrerMixin:
    """Modified Ferrer radial light profile (Binney & Tremaine 1987).

    This model has a relatively flat brightness core and then a truncation. It
    is used in specialized circumstances such as fitting the bar of a galaxy.
    The functional form of the Modified Ferrer profile is defined as:

    .. math::

       I(R) = I_0 \\left(1 - \\left(\\frac{R}{r_{\\rm out}}\\right)^{2-\\beta}\\right)^{\\alpha}

    where ``rout`` is the outer truncation radius, ``alpha`` controls the steepness
    of the truncation, ``beta`` controls the shape, and ``I0`` is the intensity at
    the center of the profile.

    :param rout: Outer truncation radius in arcseconds.
    :param alpha: Inner slope parameter.
    :param beta: Outer slope parameter.
    :param I0: Intensity at the center of the profile in flux/arcsec^2
    """

    _model_type = "ferrer"
    _parameter_specs = {
        "rout": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "Outer truncation radius in arcseconds.",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (),
            "dynamic": True,
            "description": "Inner slope parameter.",
        },
        "beta": {
            "units": "unitless",
            "valid": (0, 1.9),
            "shape": (),
            "dynamic": True,
            "description": "Outer slope parameter.",
        },
        "I0": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "Intensity at the center of the profile in flux/arcsec^2",
        },
    }

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
    def radial_model(
        self, R: ArrayLike, rout: ArrayLike, alpha: ArrayLike, beta: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.ferrer(R, rout, alpha, beta, I0)


class iFerrerMixin:
    """Modified Ferrer radial light profile (Binney & Tremaine 1987).

    This model has a relatively flat brightness core and then a truncation. It
    is used in specialized circumstances such as fitting the bar of a galaxy.
    The functional form of the Modified Ferrer profile is defined as:

    .. math::

       I(R) = I_0 \\left(1 - \\left(\\frac{R}{r_{\\rm out}}\\right)^{2-\\beta}\\right)^{\\alpha}

    where ``rout`` is the outer truncation radius, ``alpha`` controls the steepness
    of the truncation, ``beta`` controls the shape, and ``I0`` is the intensity at
    the center of the profile.

    ``rout``, ``alpha``, ``beta``, and ``I0`` are batched by their first dimension,
    allowing for multiple Ferrer profiles to be defined at once.

    :param rout: Outer truncation radius in arcseconds.
    :param alpha: Inner slope parameter.
    :param beta: Outer slope parameter.
    :param I0: Intensity at the center of the profile in flux/arcsec^2
    """

    _model_type = "ferrer"
    _parameter_specs = {
        "rout": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "Outer truncation radius in arcseconds.",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (None,),
            "dynamic": True,
            "description": "Inner slope parameter.",
        },
        "beta": {
            "units": "unitless",
            "valid": (0, 1.9),
            "shape": (None,),
            "dynamic": True,
            "description": "Outer slope parameter.",
        },
        "I0": {
            "units": "flux/arcsec^2",
            "valid": (0.0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "Intensity at the center of the profile in flux/arcsec^2",
        },
    }

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
    def iradial_model(
        self,
        i: int,
        R: ArrayLike,
        rout: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        I0: ArrayLike,
    ) -> ArrayLike:
        return func.ferrer(R, rout[i], alpha[i], beta[i], I0[i])


class FerrerPSFMixin:
    """Modified Ferrer radial light profile (Binney & Tremaine 1987).

    This model has a relatively flat brightness core and then a truncation. It
    is used in specialized circumstances such as fitting the bar of a galaxy.
    The functional form of the Modified Ferrer profile is defined as:

    .. math::

       I(R) = I_0 \\left(1 - \\left(\\frac{R}{r_{\\rm out}}\\right)^{2-\\beta}\\right)^{\\alpha}

    where ``rout`` is the outer truncation radius, ``alpha`` controls the steepness
    of the truncation, ``beta`` controls the shape, and ``I0`` is the intensity at
    the center of the profile.

    :param rout: Outer truncation radius in pixels.
    :param alpha: Inner slope parameter.
    :param beta: Outer slope parameter.
    :param I0: Intensity at the center of the profile in flux/pix^2
    """

    _model_type = "ferrer"
    _parameter_specs = {
        "rout": {
            "units": "pix",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "Outer truncation radius in pixels.",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (),
            "dynamic": True,
            "description": "Inner slope parameter.",
        },
        "beta": {
            "units": "unitless",
            "valid": (0, 1.9),
            "shape": (),
            "dynamic": True,
            "description": "Outer slope parameter.",
        },
        "I0": {
            "units": "flux/pix^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": False,
            "value": 1.0,
            "description": "Intensity at the center of the profile in flux/pix^2",
        },
    }

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
    def radial_model(
        self, R: ArrayLike, rout: ArrayLike, alpha: ArrayLike, beta: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.ferrer(R, rout, alpha, beta, I0)
