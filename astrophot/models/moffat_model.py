from caskade import forward

from .galaxy_model_object import GalaxyModel
from .psf_model_object import PSFModel
from .warp_model import WarpGalaxy
from .ray_model import RayGalaxy
from .wedge_model import WedgeGalaxy
from .superellipse_model import SuperEllipseGalaxy
from .foureirellipse_model import FourierEllipseGalaxy
from ..utils.conversions.functions import moffat_I0_to_flux
from .mixins import MoffatMixin, InclinedMixin, RadialMixin

__all__ = ("MoffatGalaxy", "MoffatPSF")


class MoffatGalaxy(MoffatMixin, RadialMixin, GalaxyModel):
    """basic galaxy model with a Moffat profile for the radial light
    profile. The functional form of the Moffat profile is defined as:

    I(R) = I0 / (1 + (R/Rd)^2)^n

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, I0 is the central flux
    density, Rd is the scale length for the profile, and n is the
    concentration index which controls the shape of the profile.

    Parameters:
        n: Concentration index which controls the shape of the brightness profile
        I0: brightness at the center of the profile, represented as the log of the brightness divided by pixel scale squared.
        Rd: scale length radius

    """

    usable = True

    @forward
    def total_flux(self, n, Rd, I0, q):
        return moffat_I0_to_flux(I0, n, Rd, q)


class MoffatPSF(MoffatMixin, RadialMixin, PSFModel):
    """basic point source model with a Moffat profile for the radial light
    profile. The functional form of the Moffat profile is defined as:

    I(R) = I0 / (1 + (R/Rd)^2)^n

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, I0 is the central flux
    density, Rd is the scale length for the profile, and n is the
    concentration index which controls the shape of the profile.

    Parameters:
        n: Concentration index which controls the shape of the brightness profile
        I0: brightness at the center of the profile, represented as the log of the brightness divided by pixel scale squared.
        Rd: scale length radius

    """

    usable = True

    @forward
    def total_flux(self, n, Rd, I0):
        return moffat_I0_to_flux(I0, n, Rd, 1.0)


class Moffat2DPSF(InclinedMixin, MoffatPSF):

    _model_type = "2d"
    usable = True

    @forward
    def total_flux(self, n, Rd, I0, q):
        return moffat_I0_to_flux(I0, n, Rd, q)


class MoffatSuperEllipseGalaxy(MoffatMixin, RadialMixin, SuperEllipseGalaxy):
    usable = True


class MoffatFourierEllipseGalaxy(MoffatMixin, RadialMixin, FourierEllipseGalaxy):
    usable = True


class MoffatWarpGalaxy(MoffatMixin, RadialMixin, WarpGalaxy):
    usable = True


class MoffatWedgeGalaxy(MoffatMixin, WedgeGalaxy):
    usable = True


class MoffatRayGalaxy(MoffatMixin, RayGalaxy):
    usable = True
