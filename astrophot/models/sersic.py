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
    def total_flux(self, Ie, n, Re, q, window=None):
        return sersic_Ie_to_flux_torch(Ie, n, Re, q)


@combine_docstrings
class TSersicGalaxy(TruncationMixin, SersicMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SersicPSF(SersicMixin, RadialMixin, PSFModel):
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
