import numpy as np
import torch

from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .star_model_object import Star_Model
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from ._shared_methods import spline_segment_initialize, select_target
from ..utils.decorators import ignore_numpy_warnings, default_internal

__all__ = [
    "Spline_Galaxy",
    "Spline_Star",
    "Spline_Warp",
    "Spline_SuperEllipse",
    "Spline_FourierEllipse",
    "Spline_Ray",
    "Spline_SuperEllipse_Warp",
    "Spline_FourierEllipse_Warp",
]


# First Order
######################################################################
class Spline_Galaxy(Galaxy_Model):
    """Basic galaxy model with a spline radial light profile. The
    light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {Galaxy_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


class Spline_Star(Star_Model):
    """star model with a spline radial light profile. The light
    profile is defined as a cubic spline interpolation of the stored
    brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {Star_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    @default_internal
    def transform_coordinates(self, X=None, Y=None, image=None, parameters=None):
        return X, Y

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None):
        return self.radial_model(
            self.radius_metric(X, Y, image, parameters), image, parameters
        )

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


class Spline_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a spline light
    profile. The light profile is defined as a cubic spline
    interpolation of the stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {Warp_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


# Second Order
######################################################################
class Spline_SuperEllipse(SuperEllipse_Galaxy):
    """The light profile is defined as a cubic spline interpolation of
    the stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


class Spline_FourierEllipse(FourierEllipse_Galaxy):
    """The light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


class Spline_Ray(Ray_Galaxy):
    """ray galaxy model with a spline light profile. The light
    profile is defined as a cubic spline interpolation of the stored
    brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): 2D Tensor of brighntess values for each ray, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {Ray_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        spline_segment_initialize(
            self,
            target=target,
            parameters=parameters,
            segments=self.rays,
            symmetric=self.symmetric_rays,
        )

    from ._shared_methods import spline_iradial_model as iradial_model


class Spline_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a spline light profile. The light
    profile is defined as a cubic spline interpolation of the stored
    brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): 2D Tensor of brighntess values for each wedge, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        spline_segment_initialize(
            self,
            target=target,
            parameters=parameters,
            segments=self.wedges,
            symmetric=self.symmetric_wedges,
        )

    from ._shared_methods import spline_iradial_model as iradial_model


# Third Order
######################################################################
class Spline_SuperEllipse_Warp(SuperEllipse_Warp):
    """The light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model


class Spline_FourierEllipse_Warp(FourierEllipse_Warp):
    """The light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I(R): Tensor of brighntess values, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"spline {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("I(R)",)
    useable = True
    extend_profile = True

    from ._shared_methods import spline_initialize as initialize
    from ._shared_methods import spline_radial_model as radial_model
