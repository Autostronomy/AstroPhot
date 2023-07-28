import torch
import numpy as np
from scipy.stats import iqr
from scipy.optimize import minimize

from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from .star_model_object import Star_Model
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from ._shared_methods import (
    parametric_initialize,
    parametric_segment_initialize,
    select_target,
)
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.initialize import isophotes
from ..utils.parametric_profiles import sersic_torch, sersic_np
from ..utils.conversions.coordinates import Rotate_Cartesian
from ..utils.conversions.functions import sersic_Ie_to_flux_torch, general_uncertainty_prop


__all__ = [
    "Sersic_Galaxy",
    "Sersic_Star",
    "Sersic_Warp",
    "Sersic_SuperEllipse",
    "Sersic_FourierEllipse",
    "Sersic_Ray",
    "Sersic_Wedge",
    "Sersic_SuperEllipse_Warp",
    "Sersic_FourierEllipse_Warp",
]


def _x0_func(model, R, F):
    return 2.0, R[4], F[4]


def _wrap_sersic(R, n, r, i):
    return sersic_np(R, n, r, 10 ** (i))


class Sersic_Galaxy(Galaxy_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
        "Ie": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    @default_internal
    def total_flux(self, parameters=None):
        return sersic_Ie_to_flux_torch(
            10 ** parameters["Ie"].value,
            parameters["n"].value,
            parameters["Re"].value,
            parameters["q"].value,
        )
    @default_internal
    def total_flux_uncertainty(self, parameters=None):
        return general_uncertainty_prop(
            (10 ** parameters["Ie"].value,
             parameters["n"].value,
             parameters["Re"].value,
             parameters["q"].value
            ),
            ((10 ** parameters["Ie"].value) * parameters["Ie"].uncertainty * torch.log(10 * torch.ones_like(parameters["Ie"].value)),
             parameters["n"].uncertainty,
             parameters["Re"].uncertainty,
             parameters["q"].uncertainty
            ),
            sersic_Ie_to_flux_torch
        )
        # return sersic_Ie_to_flux_uncertainty_torch(
        #     10 ** parameters["Ie"].value,
        #     parameters["n"].value,
        #     parameters["Re"].value,
        #     parameters["q"].value,
        #     (10 ** parameters["Ie"].value) * parameters["Ie"].uncertainty * torch.log(10 * torch.ones_like(parameters["Ie"].value)),
        #     parameters["n"].uncertainty,
        #     parameters["Re"].uncertainty,
        #     parameters["q"].uncertainty,
        # )

    def _integrate_reference(self, image_data, image_header, parameters):
        tot = self.total_flux(parameters)
        return tot / image_data.numel()

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_Star(Star_Model):
    """basic star model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {Star_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Star_Model._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model

    @default_internal
    def total_flux(self, parameters=None):
        return sersic_Ie_to_flux_torch(
            10 ** parameters["Ie"].value,
            parameters["n"].value,
            parameters["Re"].value,
            torch.ones_like(parameters["n"].value),
        )

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


class Sersic_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a sersic profile for the radial
    light profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a sersic profile for the
    radial light profile. The functional form of the Sersic profile is
    defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a sersic
    profile for the radial light profile. The functional form of the
    Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a sersic
    profile for the radial light profile. The functional form of the
    Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a sersic profile for the radial
    light model. The functional form of the Sersic profile is defined
    as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_initialize(
            self, parameters, target, _wrap_sersic, ("n", "Re", "Ie"), _x0_func
        )

    from ._shared_methods import sersic_radial_model as radial_model


class Sersic_Ray(Ray_Galaxy):
    """ray galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("n", "Re", "Ie")
    useable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        parametric_segment_initialize(
            model=self,
            target=target,
            parameters=parameters,
            prof_func=_wrap_sersic,
            params=("n", "Re", "Ie"),
            x0_func=_x0_func,
            segments=self.rays,
        )

    from ._shared_methods import sersic_iradial_model as iradial_model


class Sersic_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    model_type = f"sersic {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36, 8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0, None)},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + ("n", "Re", "Ie")
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
            prof_func=_wrap_sersic,
            params=("n", "Re", "Ie"),
            x0_func=_x0_func,
            segments=self.wedges,
        )

    from ._shared_methods import sersic_iradial_model as iradial_model
