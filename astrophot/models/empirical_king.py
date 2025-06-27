from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    EmpiricalKingMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iEmpiricalKingMixin,
)

__all__ = (
    "EmpiricalKingGalaxy",
    "EmpiricalKingPSF",
    "EmpiricalKingSuperEllipse",
    "EmpiricalKingFourierEllipse",
    "EmpiricalKingWarp",
    "EmpiricalKingRay",
    "EmpiricalKingWedge",
)


class EmpiricalKingGalaxy(EmpiricalKingMixin, RadialMixin, GalaxyModel):
    usable = True


class EmpiricalKingPSF(EmpiricalKingMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


class EmpiricalKingSuperEllipse(EmpiricalKingMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class EmpiricalKingFourierEllipse(EmpiricalKingMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class EmpiricalKingWarp(EmpiricalKingMixin, WarpMixin, GalaxyModel):
    usable = True


class EmpiricalKingRay(iEmpiricalKingMixin, RayMixin, GalaxyModel):
    usable = True


class EmpiricalKingWedge(iEmpiricalKingMixin, WedgeMixin, GalaxyModel):
    usable = True
