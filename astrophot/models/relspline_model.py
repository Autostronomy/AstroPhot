import numpy as np
import torch

from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .star_model_object import Star_Model
from .ray_model import Ray_Galaxy
from .wedge_model import Wedge_Galaxy
from ..utils.decorators import ignore_numpy_warnings, default_internal

__all__ = [
    "RelSpline_Galaxy",
    "RelSpline_Star",
]


# First Order
######################################################################
class RelSpline_Galaxy(Galaxy_Model):
    """Basic galaxy model with a spline radial light profile. The
    light profile is defined as a cubic spline interpolation of the
    stored brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I0: Central brightness
        dI(R): Tensor of brighntess values relative to central brightness, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"relspline {Galaxy_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "dI(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("I0", "dI(R)")
    useable = True
    extend_profile = True

    from ._shared_methods import relspline_initialize as initialize
    from ._shared_methods import relspline_radial_model as radial_model


class RelSpline_Star(Star_Model):
    """star model with a spline radial light profile. The light
    profile is defined as a cubic spline interpolation of the stored
    brightness values:

    I(R) = interp(R, profR, I)

    where I(R) is the brightness along the semi-major axis, interp is
    a cubic spline function, R is the semi-major axis length, profR is
    a list of radii for the spline, I is a corresponding list of
    brightnesses at each profR value.

    Parameters:
        I0: Central brightness
        dI(R): Tensor of brighntess values relative to central brightness, represented as the log of the brightness divided by pixelscale squared

    """

    model_type = f"relspline {Star_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "dI(R)": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("I0", "dI(R)")
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

    from ._shared_methods import relspline_initialize as initialize
    from ._shared_methods import relspline_radial_model as radial_model
