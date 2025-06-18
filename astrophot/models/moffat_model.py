from caskade import forward

from .galaxy_model_object import Galaxy_Model
from .psf_model_object import PSF_Model
from ..utils.conversions.functions import moffat_I0_to_flux
from .mixins import MoffatMixin, InclinedMixin

__all__ = ["Moffat_Galaxy", "Moffat_PSF"]


class Moffat_Galaxy(MoffatMixin, Galaxy_Model):
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


class Moffat_PSF(MoffatMixin, PSF_Model):
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
    model_integrated = False

    @forward
    def total_flux(self, n, Rd, I0):
        return moffat_I0_to_flux(I0, n, Rd, 1.0)


class Moffat2D_PSF(InclinedMixin, Moffat_PSF):

    _model_type = "2d"
    usable = True
    model_integrated = False

    @forward
    def total_flux(self, n, Rd, I0, q):
        return moffat_I0_to_flux(I0, n, Rd, q)
