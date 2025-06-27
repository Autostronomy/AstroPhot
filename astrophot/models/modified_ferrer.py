from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    ModifiedFerrerMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iModifiedFerrerMixin,
)

__all__ = (
    "ModifiedFerrerGalaxy",
    "ModifiedFerrerPSF",
    "ModifiedFerrerSuperEllipse",
    "ModifiedFerrerFourierEllipse",
    "ModifiedFerrerWarp",
    "ModifiedFerrerRay",
    "ModifiedFerrerWedge",
)


class ModifiedFerrerGalaxy(ModifiedFerrerMixin, RadialMixin, GalaxyModel):
    usable = True


class ModifiedFerrerPSF(ModifiedFerrerMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


class ModifiedFerrerSuperEllipse(ModifiedFerrerMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class ModifiedFerrerFourierEllipse(ModifiedFerrerMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class ModifiedFerrerWarp(ModifiedFerrerMixin, WarpMixin, GalaxyModel):
    usable = True


class ModifiedFerrerRay(iModifiedFerrerMixin, RayMixin, GalaxyModel):
    usable = True


class ModifiedFerrerWedge(iModifiedFerrerMixin, WedgeMixin, GalaxyModel):
    usable = True
