import torch
import numpy as np

from .galaxy_model_object import Galaxy_Model
from .star_model_object import Star_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from ._shared_methods import (
    parametric_initialize,
    parametric_segment_initialize,
    select_target,
)
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.parametric_profiles import nuker_np

__all__ = [
    "Nuker_Galaxy",
    "Nuker_Star",
    "Nuker_SuperEllipse",
    "Nuker_SuperEllipse_Warp",
    "Nuker_FourierEllipse",
    "Nuker_FourierEllipse_Warp",
    "Nuker_Warp",
    "Nuker_Ray",
]


def _x0_func(model_params, R, F):
    return R[4], F[4], 1.0, 2.0, 0.5


def _wrap_nuker(R, rb, ib, a, b, g):
    return nuker_np(R, rb, 10 ** (ib), a, b, g)


class Nuker_Galaxy(Galaxy_Model):
    """basic galaxy model with a Nuker profile for the radial light
    profile. The functional form of the Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {Galaxy_Model.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Galaxy_Model._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_Star(Star_Model):
    """basic star model with a Nuker profile for the radial light
    profile. The functional form of the Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {Star_Model.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Star_Model._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        return self.radial_model(
            self.radius_metric(X, Y, image=image, parameters=parameters),
            image=image,
            parameters=parameters,
        )


class Nuker_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a Nuker profile for the radial
    light profile. The functional form of the Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a Nuker profile for the
    radial light profile. The functional form of the Nuker profile is
    defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope


    """

    model_type = f"nuker {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a Nuker
    profile for the radial light profile. The functional form of the
    Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a Nuker
    profile for the radial light profile. The functional form of the
    Nuker profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a Nuker profile for the radial
    light model. The functional form of the Nuker profile is defined
    as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Warp_Galaxy._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self,
            parameters,
            target,
            _wrap_nuker,
            ("Rb", "Ib", "alpha", "beta", "gamma"),
            _x0_func,
        )

    from ._shared_methods import nuker_radial_model as radial_model


class Nuker_Ray(Ray_Galaxy):
    """ray galaxy model with a nuker profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Ray_Galaxy._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_segment_initialize(
            model=self,
            parameters=parameters,
            target=target,
            prof_func=_wrap_nuker,
            params=("Rb", "Ib", "alpha", "beta", "gamma"),
            x0_func=_x0_func,
            segments=self.rays,
        )

    from ._shared_methods import nuker_iradial_model as iradial_model


class Nuker_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a nuker profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ib * 2^((beta-gamma)/alpha) * (R / Rb)^(-gamma) * (1 + (R/Rb)^alpha)^((gamma - beta)/alpha)

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ib is the flux density at
    the scale radius Rb, Rb is the scale length for the profile, beta
    is the outer power law slope, gamma is the iner power law slope,
    and alpha is the sharpness of the transition.

    Parameters:
        Ib: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
        Rb: scale length radius
        alpha: sharpness of transition between power law slopes
        beta: outer power law slope
        gamma: inner power law slope

    """

    model_type = f"nuker {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "Rb": {"units": "arcsec", "limits": (0, None)},
        "Ib": {"units": "log10(flux/arcsec^2)"},
        "alpha": {"units": "none", "limits": (0, None)},
        "beta": {"units": "none", "limits": (0, None)},
        "gamma": {"units": "none"},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + (
        "Rb",
        "Ib",
        "alpha",
        "beta",
        "gamma",
    )
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_segment_initialize(
            model=self,
            parameters=parameters,
            target=target,
            prof_func=_wrap_nuker,
            params=("Rb", "Ib", "alpha", "beta", "gamma"),
            x0_func=_x0_func,
            segments=self.wedges,
        )

    from ._shared_methods import nuker_iradial_model as iradial_model
