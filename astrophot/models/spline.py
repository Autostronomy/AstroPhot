from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    SplineMixin,
    RadialMixin,
    iSplineMixin,
    RayMixin,
    WedgeMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)
from ..utils.decorators import combine_docstrings


__all__ = [
    "SplineGalaxy",
    "SplinePSF",
    "SplineWarp",
    "SplineSuperEllipse",
    "SplineFourierEllipse",
    "SplineRay",
    "SplineWedge",
]


@combine_docstrings
class SplineGalaxy(SplineMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SplinePSF(SplineMixin, RadialMixin, PSFModel):
    usable = True


@combine_docstrings
class SplineSuperEllipse(SplineMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SplineFourierEllipse(SplineMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SplineWarp(SplineMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SplineRay(iSplineMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class SplineWedge(iSplineMixin, WedgeMixin, GalaxyModel):
    usable = True
