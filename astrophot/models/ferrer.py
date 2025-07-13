from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    FerrerMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iFerrerMixin,
)

__all__ = (
    "FerrerGalaxy",
    "FerrerPSF",
    "FerrerSuperEllipse",
    "FerrerFourierEllipse",
    "FerrerWarp",
    "FerrerRay",
    "FerrerWedge",
)


class FerrerGalaxy(FerrerMixin, RadialMixin, GalaxyModel):
    usable = True


class FerrerPSF(FerrerMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


class FerrerSuperEllipse(FerrerMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


class FerrerFourierEllipse(FerrerMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


class FerrerWarp(FerrerMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


class FerrerRay(iFerrerMixin, RayMixin, GalaxyModel):
    usable = True


class FerrerWedge(iFerrerMixin, WedgeMixin, GalaxyModel):
    usable = True
