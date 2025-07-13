from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    KingMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iKingMixin,
)

__all__ = (
    "KingGalaxy",
    "KingPSF",
    "KingSuperEllipse",
    "KingFourierEllipse",
    "KingWarp",
    "KingRay",
    "KingWedge",
)


class KingGalaxy(KingMixin, RadialMixin, GalaxyModel):
    usable = True


class KingPSF(KingMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


class KingSuperEllipse(KingMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


class KingFourierEllipse(KingMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


class KingWarp(KingMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


class KingRay(iKingMixin, RayMixin, GalaxyModel):
    usable = True


class KingWedge(iKingMixin, WedgeMixin, GalaxyModel):
    usable = True
