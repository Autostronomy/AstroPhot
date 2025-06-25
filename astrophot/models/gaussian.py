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

__all__ = [
    "GaussianGalaxy",
    "GaussianPSF",
    "GaussianSuperEllipse",
    "GaussianFourierEllipse",
    "GaussianWarp",
    "GaussianRay",
    "GaussianWedge",
]


class GaussianGalaxy(GaussianMixin, RadialMixin, GalaxyModel):
    """Basic galaxy model with Gaussian as the radial light profile. The
    gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True


class GaussianPSF(GaussianMixin, RadialMixin, PSFModel):
    _parameter_specs = {"flux": {"units": "flux", "value": 1.0}}
    usable = True


class GaussianSuperEllipse(GaussianMixin, SuperEllipseMixin, GalaxyModel):
    usable = True


class GaussianFourierEllipse(GaussianMixin, FourierEllipseMixin, GalaxyModel):
    usable = True


class GaussianWarp(GaussianMixin, WarpMixin, GalaxyModel):
    usable = True


class GaussianRay(iGaussianMixin, RayMixin, GalaxyModel):
    usable = True


class GaussianWedge(iGaussianMixin, WedgeMixin, GalaxyModel):
    usable = True
