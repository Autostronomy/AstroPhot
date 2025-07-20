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
from ..utils.decorators import combine_docstrings


__all__ = (
    "FerrerGalaxy",
    "FerrerPSF",
    "FerrerSuperEllipse",
    "FerrerFourierEllipse",
    "FerrerWarp",
    "FerrerRay",
    "FerrerWedge",
)


@combine_docstrings
class FerrerGalaxy(FerrerMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class FerrerPSF(FerrerMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class FerrerSuperEllipse(FerrerMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class FerrerFourierEllipse(FerrerMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class FerrerWarp(FerrerMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class FerrerRay(iFerrerMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class FerrerWedge(iFerrerMixin, WedgeMixin, GalaxyModel):
    usable = True
