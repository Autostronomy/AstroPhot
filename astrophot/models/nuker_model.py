from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .warp_model import WarpGalaxy
from .ray_model import RayGalaxy
from .wedge_model import WedgeGalaxy
from .superellipse_model import SuperEllipseGalaxy
from .foureirellipse_model import FourierEllipseGalaxy
from .mixins import NukerMixin, RadialMixin

__all__ = [
    "NukerGalaxy",
    "NukerPSF",
    "NukerSuperEllipse",
    "NukerFourierEllipse",
    "NukerWarp",
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
    """basic point source model with a Nuker profile for the radial light
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


class NukerSuperEllipse(NukerMixin, RadialMixin, SuperEllipseGalaxy):
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

    usable = True


class NukerFourierEllipse(NukerMixin, RadialMixin, FourierEllipseGalaxy):
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

    usable = True


class NukerWarp(NukerMixin, RadialMixin, WarpGalaxy):
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

    usable = True


class NukerRay(iNukerMixin, RayGalaxy):
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

    usable = True


class NukerWedge(iNukerMixin, WedgeGalaxy):
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

    usable = True
