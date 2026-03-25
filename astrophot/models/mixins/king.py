import torch
import numpy as np

from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from ...utils.parametric_profiles import king_np
from .._shared_methods import parametric_initialize, parametric_segment_initialize
from .. import func


def x0_func(model_params, R, F):
    return R[2], R[5], 2, 10 ** F[0]


class KingMixin:
    """Empirical King radial light profile (Elson 1999).

    Often used for star clusters. By default the profile has ``alpha = 2`` but we
    allow the parameter to vary freely for fitting. The functional form of the
    Empirical King profile is defined as:

    .. math::

       I(R) = I_0\\left[\\frac{1}{(1 + (R/R_c)^2)^{1/\\alpha}} - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{\\alpha}\\left[1 - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{-\\alpha}

    where ``R_c`` is the core radius, ``R_t`` is the truncation radius, and ``I_0`` is
    the intensity at the center of the profile. ``alpha`` is the concentration
    index which controls the shape of the profile.

    :param Rc: core radius
    :param Rt: truncation radius
    :param alpha: concentration index which controls the shape of the brightness profile
    :param I0: intensity at the center of the profile
    """

    _model_type = "king"
    _parameter_specs = {
        "Rc": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "core radius",
        },
        "Rt": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "truncation radius",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (),
            "value": 2.0,
            "dynamic": False,
            "description": "concentration index which controls the shape of the brightness profile",
        },
        "I0": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "intensity at the center of the profile",
        },
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: king_np(r, x[0], x[1], 2.0, x[2]),
            ("Rc", "Rt", "I0"),
            x0_func,
        )

    @forward
    def radial_model(
        self, R: ArrayLike, Rc: ArrayLike, Rt: ArrayLike, alpha: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.king(R, Rc, Rt, alpha, I0)


class iKingMixin:
    """Empirical King radial light profile (Elson 1999).

    Often used for star clusters. By default the profile has ``alpha = 2`` but we
    allow the parameter to vary freely for fitting. The functional form of the
    Empirical King profile is defined as:

    .. math::

       I(R) = I_0\\left[\\frac{1}{(1 + (R/R_c)^2)^{1/\\alpha}} - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{\\alpha}\\left[1 - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{-\\alpha}

    where ``R_c`` is the core radius, ``R_t`` is the truncation radius, and ``I_0`` is
    the intensity at the center of the profile. ``alpha`` is the concentration
    index which controls the shape of the profile.

    ``Rc``, ``Rt``, ``alpha``, and ``I0`` are batched by their first dimension, allowing
    for multiple King profiles to be defined at once.

    :param Rc: core radius
    :param Rt: truncation radius
    :param alpha: concentration index which controls the shape of the brightness profile
    :param I0: intensity at the center of the profile
    """

    _model_type = "king"
    _parameter_specs = {
        "Rc": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "core radius",
        },
        "Rt": {
            "units": "arcsec",
            "valid": (0.0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "truncation radius",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (None,),
            "dynamic": False,
            "description": "concentration index which controls the shape of the brightness profile",
        },
        "I0": {
            "units": "flux/arcsec^2",
            "valid": (0, None),
            "shape": (None,),
            "dynamic": True,
            "description": "intensity at the center of the profile",
        },
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.alpha.initialized:
            self.alpha.value = 2.0 * np.ones(self.segments)

        parametric_segment_initialize(
            model=self,
            target=self.target[self.window],
            prof_func=lambda r, *x: king_np(r, x[0], x[1], 2.0, x[2]),
            params=("Rc", "Rt", "I0"),
            x0_func=x0_func,
            segments=self.segments,
        )

    @forward
    def iradial_model(
        self, i: int, R: ArrayLike, Rc: ArrayLike, Rt: ArrayLike, alpha: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.king(R, Rc[i], Rt[i], alpha[i], I0[i])


class KingPSFMixin:
    """Empirical King radial light profile (Elson 1999).

    Often used for star clusters. By default the profile has ``alpha = 2`` but we
    allow the parameter to vary freely for fitting. The functional form of the
    Empirical King profile is defined as:

    .. math::

       I(R) = I_0\\left[\\frac{1}{(1 + (R/R_c)^2)^{1/\\alpha}} - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{\\alpha}\\left[1 - \\frac{1}{(1 + (R_t/R_c)^2)^{1/\\alpha}}\\right]^{-\\alpha}

    where ``R_c`` is the core radius, ``R_t`` is the truncation radius, and ``I_0`` is
    the intensity at the center of the profile. ``alpha`` is the concentration
    index which controls the shape of the profile.

    :param Rc: core radius [pix]
    :param Rt: truncation radius [pix]
    :param alpha: concentration index which controls the shape of the brightness profile
    :param I0: intensity at the center of the profile [flux/pix^2]
    """

    _model_type = "king"
    _parameter_specs = {
        "Rc": {
            "units": "pix",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "core radius [pix]",
        },
        "Rt": {
            "units": "pix",
            "valid": (0.0, None),
            "shape": (),
            "dynamic": True,
            "description": "truncation radius [pix]",
        },
        "alpha": {
            "units": "unitless",
            "valid": (0, 10),
            "shape": (),
            "value": 2.0,
            "dynamic": False,
            "description": "concentration index which controls the shape of the brightness profile",
        },
        "I0": {
            "units": "flux/pix^2",
            "valid": (0, None),
            "shape": (),
            "dynamic": False,
            "value": 1.0,
            "description": "intensity at the center of the profile [flux/pix^2]",
        },
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        parametric_initialize(
            self,
            self.target[self.window],
            lambda r, *x: king_np(r, x[0], x[1], 2.0, x[2]),
            ("Rc", "Rt", "I0"),
            x0_func,
        )

    @forward
    def radial_model(
        self, R: ArrayLike, Rc: ArrayLike, Rt: ArrayLike, alpha: ArrayLike, I0: ArrayLike
    ) -> ArrayLike:
        return func.king(R, Rc, Rt, alpha, I0)
