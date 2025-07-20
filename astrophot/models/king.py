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
from ..utils.decorators import combine_docstrings


__all__ = (
    "KingGalaxy",
    "KingPSF",
    "KingSuperEllipse",
    "KingFourierEllipse",
    "KingWarp",
    "KingRay",
    "KingWedge",
)


@combine_docstrings
class KingGalaxy(KingMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class KingPSF(KingMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class KingSuperEllipse(KingMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class KingFourierEllipse(KingMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class KingWarp(KingMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class KingRay(iKingMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class KingWedge(iKingMixin, WedgeMixin, GalaxyModel):
    usable = True
