from .galaxy_model_object import GalaxyModel
from ..utils.decorators import combine_docstrings
from .psf_model_object import PSFModel
from .mixins import (
    ExponentialMixin,
    iExponentialMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
)

__all__ = [
    "ExponentialGalaxy",
    "ExponentialPSF",
    "ExponentialSuperEllipse",
    "ExponentialFourierEllipse",
    "ExponentialWarp",
    "ExponentialRay",
    "ExponentialWedge",
]


@combine_docstrings
class ExponentialGalaxy(ExponentialMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class ExponentialPSF(ExponentialMixin, RadialMixin, PSFModel):
    _parameter_specs = {"Ie": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class ExponentialSuperEllipse(ExponentialMixin, RadialMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


@combine_docstrings
class ExponentialFourierEllipse(ExponentialMixin, RadialMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


@combine_docstrings
class ExponentialWarp(ExponentialMixin, RadialMixin, WarpMixin, GalaxyModel):
    usable = True


@combine_docstrings
class ExponentialRay(iExponentialMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class ExponentialWedge(iExponentialMixin, WedgeMixin, GalaxyModel):
    usable = True
