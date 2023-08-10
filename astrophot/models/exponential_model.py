from typing import Optional

import torch
import numpy as np
from scipy.stats import iqr
from scipy.optimize import minimize

from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .star_model_object import Star_Model
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from ._shared_methods import (
    parametric_initialize,
    parametric_segment_initialize,
    select_target,
)
from ..utils.initialize import isophotes
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.parametric_profiles import exponential_torch, exponential_np
from ..utils.conversions.coordinates import Rotate_Cartesian

__all__ = [
    "Exponential_Galaxy",
    "Exponential_Star",
    "Exponential_SuperEllipse",
    "Exponential_SuperEllipse_Warp",
    "Exponential_Warp",
    "Exponential_Ray",
    "Exponential_Wedge",
]


def _x0_func(model_params, R, F):
    return R[4], F[4]


def _wrap_exp(R, re, ie):
    return exponential_np(R, re, 10 ** ie)


class Exponential_Galaxy(Galaxy_Model):
    """basic galaxy model with a exponential profile for the radial light
    profile. The light profile is defined as:

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {Galaxy_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target=None, parameters: Optional["Parameter_Group"] = None, **kwargs
    ):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_Star(Star_Model):
    """basic star model with a exponential profile for the radial light
    profile.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {Star_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Star_Model._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model

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


class Exponential_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a exponential profile for the radial
    light profile.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a exponential profile for the
    radial light profile.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with an
    expoential profile for the radial light profile.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a exponential
    profile for the radial light profile.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a exponential profile for the
    radial light model.

    I(R) = Ie * exp(-b1(R/Re - 1))

    where I(R) is the brightness as a function of semi-major axis, Ie
    is the brightness at the half light radius, b1 is a constant not
    involved in the fit, R is the semi-major axis, and Re is the
    effective radius.

    Parameters:
        Ie: Brightness at half light radius, represented as the log of the brightness divided by pixelscale squared. This is proportional to a surface brightness
        Re: half light radius, represented in arcsec. This parameter cannot go below zero.

    """

    model_type = f"exponential {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_exp, ("Re", "Ie"), _x0_func
        )

    from ._shared_methods import exponential_radial_model as radial_model


class Exponential_Ray(Ray_Galaxy):
    """ray galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius.

    Parameters:
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"exponential {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("Re", "Ie")
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
            prof_func=_wrap_exp,
            params=("Re", "Ie"),
            x0_func=_x0_func,
            segments=self.rays,
        )

    from ._shared_methods import exponential_iradial_model as iradial_model


class Exponential_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a exponential profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius.

    Parameters:
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"exponential {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + ("Re", "Ie")
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
            prof_func=_wrap_exp,
            params=("Re", "Ie"),
            x0_func=_x0_func,
            segments=self.wedges,
        )

    from ._shared_methods import exponential_iradial_model as iradial_model
