from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .psf_model_object import PSF_Model
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .wedge_model import Wedge_Galaxy
from .mixins import ExponentialMixin, iExponentialMixin

__all__ = [
    "Exponential_Galaxy",
    "Exponential_PSF",
    "Exponential_SuperEllipse",
    "Exponential_SuperEllipse_Warp",
    "Exponential_Warp",
    "Exponential_Ray",
    "Exponential_Wedge",
]


class Exponential_Galaxy(ExponentialMixin, Galaxy_Model):
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


class Exponential_PSF(ExponentialMixin, PSF_Model):
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
    model_integrated = False


class Exponential_SuperEllipse(ExponentialMixin, SuperEllipse_Galaxy):
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


class Exponential_SuperEllipse_Warp(ExponentialMixin, SuperEllipse_Warp):
    """super ellipse warp galaxy model with a exponential profile for the
    radial light profile.

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


class Exponential_FourierEllipse(ExponentialMixin, FourierEllipse_Galaxy):
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


class Exponential_FourierEllipse_Warp(ExponentialMixin, FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a exponential
    profile for the radial light profile.

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


class Exponential_Warp(ExponentialMixin, Warp_Galaxy):
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


class Exponential_Ray(iExponentialMixin, Ray_Galaxy):
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


class Exponential_Wedge(iExponentialMixin, Wedge_Galaxy):
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
