from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    NukerMixin,
    RadialMixin,
    iNukerMixin,
    RayMixin,
    WedgeMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)

__all__ = [
    "NukerGalaxy",
    "NukerPSF",
    "NukerSuperEllipse",
    "NukerFourierEllipse",
    "NukerWarp",
    "NukerWedge",
    "NukerRay",
]


class NukerGalaxy(NukerMixin, RadialMixin, GalaxyModel):
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

    usable = True


class NukerPSF(NukerMixin, RadialMixin, PSFModel):
    _parameter_specs = {"Ib": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


class NukerSuperEllipse(NukerMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class NukerFourierEllipse(NukerMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class NukerWarp(NukerMixin, WarpMixin, GalaxyModel):
    usable = True


class NukerRay(iNukerMixin, RayMixin, GalaxyModel):
    usable = True


class NukerWedge(iNukerMixin, WedgeMixin, GalaxyModel):
    usable = True
