from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .mixins import (
    MoffatMixin,
    InclinedMixin,
    RadialMixin,
    WedgeMixin,
    RayMixin,
    SuperEllipseMixin,
    FourierEllipseMixin,
    WarpMixin,
    iMoffatMixin,
)
from ..utils.decorators import combine_docstrings


__all__ = (
    "MoffatGalaxy",
    "MoffatPSF",
    "Moffat2DPSF",
    "MoffatSuperEllipse",
    "MoffatFourierEllipse",
    "MoffatWarp",
    "MoffatRay",
    "MoffatWedge",
)


@combine_docstrings
class MoffatGalaxy(MoffatMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class MoffatPSF(MoffatMixin, RadialMixin, PSFModel):
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class Moffat2DPSF(MoffatMixin, InclinedMixin, RadialMixin, PSFModel):
    _model_type = "2d"
    _parameter_specs = {"I0": {"units": "flux/arcsec^2", "value": 1.0}}
    usable = True


@combine_docstrings
class MoffatSuperEllipse(MoffatMixin, SuperEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class MoffatFourierEllipse(MoffatMixin, FourierEllipseMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class MoffatWarp(MoffatMixin, WarpMixin, RadialMixin, GalaxyModel):
    usable = True


@combine_docstrings
class MoffatRay(iMoffatMixin, RayMixin, GalaxyModel):
    usable = True


@combine_docstrings
class MoffatWedge(iMoffatMixin, WedgeMixin, GalaxyModel):
    usable = True
