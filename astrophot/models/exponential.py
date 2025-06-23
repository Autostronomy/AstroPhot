from .galaxy_model_object import GalaxyModel

from .psf_model_object import PSFModel
from .mixins import (
    ExponentialMixin,
    iExponentialMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)

__all__ = [
    "ExponentialGalaxy",
    "ExponentialPSF",
    "ExponentialSuperEllipse",
    "ExponentialFourierEllipse",
    "ExponentialWarp",
    "ExponentialRay",
    "ExponentialWedge",
]


class ExponentialGalaxy(ExponentialMixin, RadialMixin, GalaxyModel):
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

    usable = True


class ExponentialPSF(ExponentialMixin, RadialMixin, PSFModel):
    usable = True


class ExponentialSuperEllipse(ExponentialMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class ExponentialFourierEllipse(ExponentialMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class ExponentialWarp(ExponentialMixin, WarpMixin, GalaxyModel):
    usable = True


class ExponentialRay(iExponentialMixin, RayMixin, GalaxyModel):
    usable = True


class ExponentialWedge(iExponentialMixin, WedgeMixin, GalaxyModel):
    usable = True
