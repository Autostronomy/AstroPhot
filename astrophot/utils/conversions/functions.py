import numpy as np
import torch
from scipy.special import gamma
from torch.special import gammaln


def sersic_n_to_b(n):
    """Compute the `b(n)` for a sersic model. This factor ensures that
    the :math:`R_e` and :math:`I_e` parameters do in fact correspond
    to the half light values and not some other scale
    radius/intensity.

    """

    return (
        2 * n
        - 1 / 3
        + 4 / (405 * n)
        + 46 / (25515 * n**2)
        + 131 / (1148175 * n**3)
        - 2194697 / (30690717750 * n**4)
    )


def sersic_I0_to_flux_np(I0, n, R, q):
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the :math:`I_0,n,R_s,q` parameters which uniquely
    define the profile (:math:`I_0` is the central intensity in
    flux/arcsec^2). Note that :math:`R_s` is not the effective radius,
    but in fact the scale radius in the more straightforward sersic
    representation:

    .. math::

      I(R) = I_0e^{-(R/R_s)^{1/n}}

    Args:
      I0: central intensity (flux/arcsec^2)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)

    """
    return 2 * np.pi * I0 * q * n * R**2 * gamma(2 * n)


def sersic_flux_to_I0_np(flux, n, R, q):
    """Compute the central intensity (flux/arcsec^2) for a 2D elliptical
    sersic given the :math:`F,n,R_s,q` parameters which uniquely
    define the profile (:math:`F` is the total flux integrated to
    infinity). Note that :math:`R_s` is not the effective radius, but
    in fact the scale radius in the more straightforward sersic
    representation:

    .. math::

      I(R) = I_0e^{-(R/R_s)^{1/n}}

    Args:
      flux: total flux integrated to infinity (flux)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)

    """
    return flux / (2 * np.pi * q * n * R**2 * gamma(2 * n))


def sersic_Ie_to_flux_np(Ie, n, R, q):
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the :math:`I_e,n,R_e,q` parameters which uniquely
    define the profile (:math:`I_e` is the intensity at :math:`R_e` in
    flux/arcsec^2). Note that :math:`R_e` is the effective radius in
    the sersic representation:

    .. math::

      I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}

    Args:
      Ie: intensity at the effective radius (flux/arcsec^2)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)

    """
    bn = sersic_n_to_b(n)
    return 2 * np.pi * Ie * R**2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n)


def sersic_flux_to_Ie_np(flux, n, R, q):
    """Compute the intensity at :math:`R_e` (flux/arcsec^2) for a 2D
    elliptical sersic given the :math:`F,n,R_e,q` parameters which
    uniquely define the profile (:math:`F` is the total flux
    integrated to infinity). Note that :math:`R_e` is the effective
    radius in the sersic representation:

    .. math::

      I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}

    Args:
      flux: flux integrated to infinity (flux)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)

    """
    bn = sersic_n_to_b(n)
    return flux / (2 * np.pi * R**2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n))


def sersic_inv_np(I, n, Re, Ie):
    """Invert the sersic profile. Compute the radius corresponding to a
    given intensity for a pure sersic profile.

    """
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * np.log(I / Ie)) ** (n))


def sersic_I0_to_flux_torch(I0, n, R, q):
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the :math:`I_0,n,R_s,q` parameters which uniquely
    define the profile (:math:`I_0` is the central intensity in
    flux/arcsec^2). Note that :math:`R_s` is not the effective radius,
    but in fact the scale radius in the more straightforward sersic
    representation:

    .. math::

      I(R) = I_0e^{-(R/R_s)^{1/n}}

    Args:
      I0: central intensity (flux/arcsec^2)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)


    """
    return 2 * np.pi * I0 * q * n * R**2 * torch.exp(gammaln(2 * n))


def sersic_flux_to_I0_torch(flux, n, R, q):
    """Compute the central intensity (flux/arcsec^2) for a 2D elliptical
    sersic given the :math:`F,n,R_s,q` parameters which uniquely
    define the profile (:math:`F` is the total flux integrated to
    infinity). Note that :math:`R_s` is not the effective radius, but
    in fact the scale radius in the more straightforward sersic
    representation:

    .. math::

      I(R) = I_0e^{-(R/R_s)^{1/n}}

    Args:
      flux: total flux integrated to infinity (flux)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)


    """
    return flux / (2 * np.pi * q * n * R**2 * torch.exp(gammaln(2 * n)))


def sersic_Ie_to_flux_torch(Ie, n, R, q):
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the :math:`I_e,n,R_e,q` parameters which uniquely
    define the profile (:math:`I_e` is the intensity at :math:`R_e` in
    flux/arcsec^2). Note that :math:`R_e` is the effective radius in
    the sersic representation:

    .. math::

      I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}

    Args:
      Ie: intensity at the effective radius (flux/arcsec^2)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)


    """
    bn = sersic_n_to_b(n)
    return (
        2 * np.pi * Ie * R**2 * q * n * (torch.exp(bn) * bn ** (-2 * n)) * torch.exp(gammaln(2 * n))
    )


def sersic_flux_to_Ie_torch(flux, n, R, q):
    """Compute the intensity at :math:`R_e` (flux/arcsec^2) for a 2D
    elliptical sersic given the :math:`F,n,R_e,q` parameters which
    uniquely define the profile (:math:`F` is the total flux
    integrated to infinity). Note that :math:`R_e` is the effective
    radius in the sersic representation:

    .. math::

      I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}

    Args:
      flux: flux integrated to infinity (flux)
      n: sersic index
      R: Scale radius
      q: axis ratio (b/a)


    """
    bn = sersic_n_to_b(n)
    return flux / (
        2 * np.pi * R**2 * q * n * (torch.exp(bn) * bn ** (-2 * n)) * torch.exp(gammaln(2 * n))
    )


def sersic_inv_torch(I, n, Re, Ie):
    """Invert the sersic profile. Compute the radius corresponding to a
    given intensity for a pure sersic profile.

    """
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * torch.log(I / Ie)) ** (n))


def moffat_I0_to_flux(I0, n, rd, q):
    """
    Compute the total flux integrated to infinity for a moffat profile.

    Args:
      I0: central intensity (flux/arcsec^2)
      n: moffat curvature parameter (unitless)
      rd: scale radius
      q: axis ratio
    """
    return I0 * np.pi * rd**2 * q / (n - 1)
