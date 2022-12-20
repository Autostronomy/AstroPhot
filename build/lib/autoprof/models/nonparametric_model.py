from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .star_model_object import Star_Model
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
import numpy as np
import torch

__all__ = [
    "NonParametric_Galaxy", "NonParametric_Star", "NonParametric_Warp",
    "NonParametric_SuperEllipse", "NonParametric_FourierEllipse", "NonParametric_Ray",
    "NonParametric_SuperEllipse_Warp", "NonParametric_FourierEllipse_Warp"
]

# First Order
######################################################################
class NonParametric_Galaxy(Galaxy_Model):
    """Basic galaxy model with a nonparametric radial light profile. The
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
    model_type = f"nonparametric {Galaxy_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Star(Star_Model):
    """star model with a nonparametric radial light profile. The light
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
    model_type = f"nonparametric {Star_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("I(R)",)
    
    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a nonparametric light
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
    model_type = f"nonparametric {Warp_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

# Second Order
######################################################################
class NonParametric_SuperEllipse(SuperEllipse_Galaxy):
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
    
    model_type = f"nonparametric {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_FourierEllipse(FourierEllipse_Galaxy):
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
    model_type = f"nonparametric {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_Ray(Ray_Galaxy):
    """ray galaxy model with a nonparametric light profile. The light
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
    model_type = f"nonparametric {Ray_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize # fixme specialized initialize
    from ._shared_methods import nonparametric_iradial_model as iradial_model

class NonParametric_Wedge(Wedge_Galaxy):
    """wedge galaxy model with a nonparametric light profile. The light
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
    model_type = f"nonparametric {Wedge_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Wedge_Galaxy._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize # fixme specialized initialize
    from ._shared_methods import nonparametric_iradial_model as iradial_model
    
# Third Order
######################################################################
class NonParametric_SuperEllipse_Warp(SuperEllipse_Warp):
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
    model_type = f"nonparametric {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model

class NonParametric_FourierEllipse_Warp(FourierEllipse_Warp):
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
    model_type = f"nonparametric {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("I(R)",)

    from ._shared_methods import nonparametric_initialize as initialize
    from ._shared_methods import nonparametric_radial_model as radial_model



