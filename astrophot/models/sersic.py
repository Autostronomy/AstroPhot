from ..param import forward
from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from ..utils.conversions.functions import sersic_Ie_to_flux_torch
from ..utils.decorators import combine_docstrings
from .mixins import (
    SersicMixin,
    RadialMixin,
    WedgeMixin,
    iSersicMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    TruncationMixin,
)

__all__ = [
    "SersicGalaxy",
    "TSersicGalaxy",
    "SersicPSF",
    "Sersic_Warp",
    "Sersic_SuperEllipse",
    "Sersic_FourierEllipse",
    "Sersic_Ray",
    "Sersic_Wedge",
]


@combine_docstrings
class SersicGalaxy(SersicMixin, RadialMixin, GalaxyModel):
    usable = True

    @forward
    def total_flux(self, Ie, n, Re, q):
        return sersic_Ie_to_flux_torch(Ie, n, Re, q)


@combine_docstrings
class TSersicGalaxy(TruncationMixin, SersicMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
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

    _parameter_specs = {"Ie": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True

    @forward
    def total_flux(self, Ie, n, Re):
        return sersic_Ie_to_flux_torch(Ie, n, Re, 1.0)


@combine_docstrings
class SersicSuperEllipse(SersicMixin, RadialMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SersicFourierEllipse(SersicMixin, RadialMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SersicWarp(SersicMixin, RadialMixin, WarpMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SersicRay(iSersicMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SersicWedge(iSersicMixin, WedgeMixin, GalaxyModel):
    usable = True
