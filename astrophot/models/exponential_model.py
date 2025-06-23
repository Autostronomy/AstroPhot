from .galaxy_model_object import GalaxyModel

from .warp_model import WarpGalaxy
from .ray_model import RayGalaxy
from .psf_model_object import PSFModel
from .superellipse_model import SuperEllipseGalaxy  # , SuperEllipse_Warp
from .foureirellipse_model import FourierEllipseGalaxy  # , FourierEllipse_Warp
from .wedge_model import WedgeGalaxy
from .mixins import ExponentialMixin, iExponentialMixin, RadialMixin

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
    """basic point source model with a exponential profile for the radial light
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

    usable = True


class ExponentialSuperEllipse(ExponentialMixin, RadialMixin, SuperEllipseGalaxy):
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

    usable = True


class ExponentialFourierEllipse(ExponentialMixin, RadialMixin, FourierEllipseGalaxy):
    """fourier mode perturbations to ellipse galaxy model with an
    exponential profile for the radial light profile.

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


class ExponentialWarp(ExponentialMixin, RadialMixin, WarpGalaxy):
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

    usable = True


class ExponentialRay(iExponentialMixin, RayGalaxy):
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

    usable = True


class ExponentialWedge(iExponentialMixin, WedgeGalaxy):
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

    usable = True
