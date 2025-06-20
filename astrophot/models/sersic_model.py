from ..param import forward
from .galaxy_model_object import GalaxyModel

# from .warp_model import Warp_Galaxy
# from .ray_model import Ray_Galaxy
# from .wedge_model import Wedge_Galaxy
from .psf_model_object import PSFModel

# from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
# from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from ..utils.conversions.functions import sersic_Ie_to_flux_torch
from .mixins import SersicMixin, RadialMixin, iSersicMixin

__all__ = [
    "SersicGalaxy",
    "SersicPSF",
    # "Sersic_Warp",
    # "Sersic_SuperEllipse",
    # "Sersic_FourierEllipse",
    # "Sersic_Ray",
    # "Sersic_Wedge",
    # "Sersic_SuperEllipse_Warp",
    # "Sersic_FourierEllipse_Warp",
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


# class Sersic_SuperEllipse(SersicMixin, SuperEllipse_Galaxy):
#     """super ellipse galaxy model with a sersic profile for the radial
#     light profile. The functional form of the Sersic profile is defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_SuperEllipse_Warp(SersicMixin, SuperEllipse_Warp):
#     """super ellipse warp galaxy model with a sersic profile for the
#     radial light profile. The functional form of the Sersic profile is
#     defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_FourierEllipse(SersicMixin, FourierEllipse_Galaxy):
#     """fourier mode perturbations to ellipse galaxy model with a sersic
#     profile for the radial light profile. The functional form of the
#     Sersic profile is defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_FourierEllipse_Warp(SersicMixin, FourierEllipse_Warp):
#     """fourier mode perturbations to ellipse galaxy model with a sersic
#     profile for the radial light profile. The functional form of the
#     Sersic profile is defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_Warp(SersicMixin, Warp_Galaxy):
#     """warped coordinate galaxy model with a sersic profile for the radial
#     light model. The functional form of the Sersic profile is defined
#     as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_Ray(iSersicMixin, Ray_Galaxy):
#     """ray galaxy model with a sersic profile for the radial light
#     model. The functional form of the Sersic profile is defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True


# class Sersic_Wedge(iSersicMixin, Wedge_Galaxy):
#     """wedge galaxy model with a sersic profile for the radial light
#     model. The functional form of the Sersic profile is defined as:

#     I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

#     where I(R) is the brightness profile as a function of semi-major
#     axis, R is the semi-major axis length, Ie is the brightness as the
#     half light radius, bn is a function of n and is not involved in
#     the fit, Re is the half light radius, and n is the sersic index
#     which controls the shape of the profile.

#     Parameters:
#         n: Sersic index which controls the shape of the brightness profile
#         Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
#         Re: half light radius

#     """

#     usable = True
