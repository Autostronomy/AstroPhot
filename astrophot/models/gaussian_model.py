from .galaxy_model_object import GalaxyModel

from .warp_model import WarpGalaxy
from .superellipse_model import SuperEllipseGalaxy
from .foureirellipse_model import FourierEllipseGalaxy
from .ray_model import RayGalaxy
from .wedge_model import WedgeGalaxy
from .psf_model_object import PSFModel
from .mixins import GaussianMixin, RadialMixin

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
    """Basic point source model with a Gaussian as the radial light profile. The
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


class GaussianSuperEllipse(GaussianMixin, RadialMixin, SuperEllipseGalaxy):
    """Super ellipse galaxy model with Gaussian as the radial light
    profile.The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True


class GaussianFourierEllipse(GaussianMixin, RadialMixin, FourierEllipseGalaxy):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile. The gaussian radial profile
    is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True


class GaussianWarp(GaussianMixin, RadialMixin, WarpGalaxy):
    """Coordinate warped galaxy model with Gaussian as the radial light
    profile. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True


class GaussianRay(iGaussianMixin, RayGalaxy):
    """ray galaxy model with a gaussian profile for the radial light
    model. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True


class GaussianWedge(iGaussianMixin, WedgeGalaxy):
    """wedge galaxy model with a gaussian profile for the radial light
    model. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """

    usable = True
