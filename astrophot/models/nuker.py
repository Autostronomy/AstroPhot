from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    NukerMixin,
    RadialMixin,
    iNukerMixin,
    RayMixin,
    WedgeMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)
from ..utils.decorators import combine_docstrings


__all__ = [
    "NukerGalaxy",
    "NukerPSF",
    "NukerSuperEllipse",
    "NukerFourierEllipse",
    "NukerWarp",
    "NukerWedge",
    "NukerRay",
]


@combine_docstrings
class NukerGalaxy(NukerMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class NukerPSF(NukerMixin, RadialMixin, PSFModel):
    _parameter_specs = {"Ib": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class NukerSuperEllipse(NukerMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class NukerFourierEllipse(NukerMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class NukerWarp(NukerMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class NukerRay(iNukerMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class NukerWedge(iNukerMixin, WedgeMixin, GalaxyModel):
    usable = True
