from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    SplineMixin,
    RadialMixin,
    iSplineMixin,
    RayMixin,
    WedgeMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)

__all__ = [
    "SplineGalaxy",
    "SplinePSF",
    "SplineWarp",
    "SplineSuperEllipse",
    "SplineFourierEllipse",
    "SplineRay",
    "SplineWedge",
]


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
    usable = True


class SplineSuperEllipse(SplineMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class SplineFourierEllipse(SplineMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class SplineWarp(SplineMixin, WarpMixin, GalaxyModel):
    usable = True


class SplineRay(iSplineMixin, RayMixin, GalaxyModel):
    usable = True


class SplineWedge(iSplineMixin, WedgeMixin, GalaxyModel):
    usable = True
