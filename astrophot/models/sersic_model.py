from ..param import forward
from .galaxy_model_object import GalaxyModel

from .warp_model import WarpGalaxy
from .ray_model import RayGalaxy
from .wedge_model import WedgeGalaxy
from .psf_model_object import PSFModel

from .superellipse_model import SuperEllipseGalaxy  # , SuperEllipse_Warp
from .foureirellipse_model import FourierEllipseGalaxy  # , FourierEllipse_Warp
from ..utils.conversions.functions import sersic_Ie_to_flux_torch
from .mixins import SersicMixin, RadialMixin, iSersicMixin

__all__ = [
    "SersicGalaxy",
    "SersicPSF",
    "Sersic_Warp",
    "Sersic_SuperEllipse",
    "Sersic_FourierEllipse",
    "Sersic_Ray",
    "Sersic_Wedge",
]


class SersicGalaxy(SersicMixin, RadialMixin, GalaxyModel):
    """basic galaxy model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True

    @forward
    def total_flux(self, Ie, n, Re, q):
        return sersic_Ie_to_flux_torch(Ie, n, Re, q)


class SersicPSF(SersicMixin, RadialMixin, PSFModel):
    """basic point source model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True

    @forward
    def total_flux(self, Ie, n, Re):
        return sersic_Ie_to_flux_torch(Ie, n, Re, 1.0)


class SersicSuperEllipse(SersicMixin, RadialMixin, SuperEllipseGalaxy):
    """super ellipse galaxy model with a sersic profile for the radial
    light profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True


class SersicFourierEllipse(SersicMixin, RadialMixin, FourierEllipseGalaxy):
    """fourier mode perturbations to ellipse galaxy model with a sersic
    profile for the radial light profile. The functional form of the
    Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True


class SersicWarp(SersicMixin, RadialMixin, WarpGalaxy):
    """warped coordinate galaxy model with a sersic profile for the radial
    light model. The functional form of the Sersic profile is defined
    as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True


class SersicRay(iSersicMixin, RayGalaxy):
    """ray galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True


class SersicWedge(iSersicMixin, WedgeGalaxy):
    """wedge galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """

    usable = True
