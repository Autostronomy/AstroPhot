from .galaxy_model_object import GalaxyModel

from .warp_model import WarpGalaxy
from .superellipse_model import SuperEllipseGalaxy  # , SuperEllipse_Warp
from .foureirellipse_model import FourierEllipseGalaxy  # , FourierEllipse_Warp
from .psf_model_object import PSFModel

from .ray_model import RayGalaxy
from .wedge_model import WedgeGalaxy
from .mixins import SplineMixin, RadialMixin

__all__ = [
    "SplineGalaxy",
    "SplinePSF",
    "SplineWarp",
    "SplineSuperEllipse",
    "SplineFourierEllipse",
    "SplineRay",
    "SplineWedge",
]


# First Order
######################################################################
class SplineGalaxy(SplineMixin, RadialMixin, GalaxyModel):
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

    usable = True


class SplinePSF(SplineMixin, RadialMixin, PSFModel):
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

    usable = True


class SplineWarp(SplineMixin, RadialMixin, WarpGalaxy):
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

    usable = True


class SplineSuperEllipse(SplineMixin, RadialMixin, SuperEllipseGalaxy):
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

    usable = True


class SplineFourierEllipse(SplineMixin, RadialMixin, FourierEllipseGalaxy):
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

    usable = True


class SplineRay(iSplineMixin, RayGalaxy):
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

    usable = True


class SplineWedge(iSplineMixin, WedgeGalaxy):
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

    usable = True
