from .galaxy_model_object import GalaxyModel

from .psf_model_object import PSFModel
from .mixins import (
    GaussianMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iGaussianMixin,
)
from ..utils.decorators import combine_docstrings


__all__ = [
    "GaussianGalaxy",
    "GaussianPSF",
    "GaussianSuperEllipse",
    "GaussianFourierEllipse",
    "GaussianWarp",
    "GaussianRay",
    "GaussianWedge",
]


@combine_docstrings
class GaussianGalaxy(GaussianMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class GaussianPSF(GaussianMixin, RadialMixin, PSFModel):
    _parameter_specs = {"flux": {"units": "flux", "value": 1.0}}
    usable = True


@combine_docstrings
class GaussianSuperEllipse(GaussianMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class GaussianFourierEllipse(GaussianMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class GaussianWarp(GaussianMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class GaussianRay(iGaussianMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class GaussianWedge(iGaussianMixin, WedgeMixin, GalaxyModel):
    usable = True
