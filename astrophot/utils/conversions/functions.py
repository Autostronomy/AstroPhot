from typing import Union
import numpy as np
from scipy.special import gamma
from ...backend_obj import backend, ArrayLike

__all__ = (
    "sersic_n_to_b",
    "sersic_I0_to_flux_np",
    "sersic_flux_to_I0_np",
    "sersic_Ie_to_flux_np",
    "sersic_flux_to_Ie_np",
    "sersic_I0_to_flux_torch",
    "sersic_flux_to_I0_torch",
    "sersic_Ie_to_flux_torch",
    "sersic_flux_to_Ie_torch",
    "sersic_inv_np",
    "sersic_inv_torch",
    "moffat_I0_to_flux",
)


def sersic_n_to_b(
    n: Union[float, np.ndarray, ArrayLike],
) -> Union[float, np.ndarray, ArrayLike]:
    """Compute the `b(n)` for a sersic model. This factor ensures that
    the $R_e$ and $I_e$ parameters do in fact correspond
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


def sersic_I0_to_flux_np(I0: np.ndarray, n: np.ndarray, R: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the $I_0,n,R_s,q$ parameters which uniquely
    define the profile ($I_0$ is the central intensity in
    flux/arcsec^2). Note that $R_s$ is not the effective radius,
    but in fact the scale radius in the more straightforward sersic
    representation:

    $$I(R) = I_0e^{-(R/R_s)^{1/n}}$$

    **Args:**
    -  `I0`: central intensity (flux/arcsec^2)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)

    """
    return 2 * np.pi * I0 * q * n * R**2 * gamma(2 * n)


def sersic_flux_to_I0_np(
    flux: np.ndarray, n: np.ndarray, R: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Compute the central intensity (flux/arcsec^2) for a 2D elliptical
    sersic given the $F,n,R_s,q$ parameters which uniquely
    define the profile ($F$ is the total flux integrated to
    infinity). Note that $R_s$ is not the effective radius, but
    in fact the scale radius in the more straightforward sersic
    representation:

    $$I(R) = I_0e^{-(R/R_s)^{1/n}}$$

    **Args:**
    -  `flux`: total flux integrated to infinity (flux)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)

    """
    return flux / (2 * np.pi * q * n * R**2 * gamma(2 * n))


def sersic_Ie_to_flux_np(Ie: np.ndarray, n: np.ndarray, R: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the $I_e,n,R_e,q$ parameters which uniquely
    define the profile ($I_e$ is the intensity at $R_e$ in
    flux/arcsec^2). Note that $R_e$ is the effective radius in
    the sersic representation:

    $$I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}$$

    **Args:**
    -  `Ie`: intensity at the effective radius (flux/arcsec^2)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)
    """
    bn = sersic_n_to_b(n)
    return 2 * np.pi * Ie * R**2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n)


def sersic_flux_to_Ie_np(
    flux: np.ndarray, n: np.ndarray, R: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Compute the intensity at $R_e$ (flux/arcsec^2) for a 2D
    elliptical sersic given the $F,n,R_e,q$ parameters which
    uniquely define the profile ($F$ is the total flux
    integrated to infinity). Note that $R_e$ is the effective
    radius in the sersic representation:

    $$I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}$$

    **Args:**
    -  `flux`: flux integrated to infinity (flux)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)

    """
    bn = sersic_n_to_b(n)
    return flux / (2 * np.pi * R**2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n))


def sersic_inv_np(I: np.ndarray, n: np.ndarray, Re: np.ndarray, Ie: np.ndarray) -> np.ndarray:
    """Invert the sersic profile. Compute the radius corresponding to a
    given intensity for a pure sersic profile.

    """
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * np.log(I / Ie)) ** (n))


def sersic_I0_to_flux_torch(I0: ArrayLike, n: ArrayLike, R: ArrayLike, q: ArrayLike) -> ArrayLike:
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the $I_0,n,R_s,q$ parameters which uniquely
    define the profile ($I_0$ is the central intensity in
    flux/arcsec^2). Note that $R_s$ is not the effective radius,
    but in fact the scale radius in the more straightforward sersic
    representation:

    $$I(R) = I_0e^{-(R/R_s)^{1/n}}$$

    **Args:**
    -  `I0`: central intensity (flux/arcsec^2)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)


    """
    return 2 * np.pi * I0 * q * n * R**2 * backend.exp(backend.gammaln(2 * n))


def sersic_flux_to_I0_torch(flux: ArrayLike, n: ArrayLike, R: ArrayLike, q: ArrayLike) -> ArrayLike:
    """Compute the central intensity (flux/arcsec^2) for a 2D elliptical
    sersic given the $F,n,R_s,q$ parameters which uniquely
    define the profile ($F$ is the total flux integrated to
    infinity). Note that $R_s$ is not the effective radius, but
    in fact the scale radius in the more straightforward sersic
    representation:

    $$I(R) = I_0e^{-(R/R_s)^{1/n}}$$

    **Args:**
    -  `flux`: total flux integrated to infinity (flux)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)

    """
    return flux / (2 * np.pi * q * n * R**2 * backend.exp(backend.gammaln(2 * n)))


def sersic_Ie_to_flux_torch(Ie: ArrayLike, n: ArrayLike, R: ArrayLike, q: ArrayLike) -> ArrayLike:
    """Compute the total flux integrated to infinity for a 2D elliptical
    sersic given the $I_e,n,R_e,q$ parameters which uniquely
    define the profile ($I_e$ is the intensity at $R_e$ in
    flux/arcsec^2). Note that $R_e$ is the effective radius in
    the sersic representation:

    $$I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}$$

    **Args:**
    -  `Ie`: intensity at the effective radius (flux/arcsec^2)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)


    """
    bn = sersic_n_to_b(n)
    return (
        2
        * np.pi
        * Ie
        * R**2
        * q
        * n
        * (backend.exp(bn) * bn ** (-2 * n))
        * backend.exp(backend.gammaln(2 * n))
    )


def sersic_flux_to_Ie_torch(flux: ArrayLike, n: ArrayLike, R: ArrayLike, q: ArrayLike) -> ArrayLike:
    """Compute the intensity at $R_e$ (flux/arcsec^2) for a 2D
    elliptical sersic given the $F,n,R_e,q$ parameters which
    uniquely define the profile ($F$ is the total flux
    integrated to infinity). Note that $R_e$ is the effective
    radius in the sersic representation:

    $$I(R) = I_ee^{-b_n[(R/R_e)^{1/n}-1]}$$

    **Args:**
    -  `flux`: flux integrated to infinity (flux)
    -  `n`: sersic index
    -  `R`: Scale radius
    -  `q`: axis ratio (b/a)


    """
    bn = sersic_n_to_b(n)
    return flux / (
        2
        * np.pi
        * R**2
        * q
        * n
        * (backend.exp(bn) * bn ** (-2 * n))
        * backend.exp(backend.gammaln(2 * n))
    )


def sersic_inv_torch(I: ArrayLike, n: ArrayLike, Re: ArrayLike, Ie: ArrayLike) -> ArrayLike:
    """Invert the sersic profile. Compute the radius corresponding to a
    given intensity for a pure sersic profile.

    """
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * backend.log(I / Ie)) ** (n))


def moffat_I0_to_flux(I0: float, n: float, rd: float, q: float) -> float:
    """
    Compute the total flux integrated to infinity for a moffat profile.

    **Args:**
    -  `I0`: central intensity (flux/arcsec^2)
    -  `n`: moffat curvature parameter (unitless)
    -  `rd`: scale radius
    -  `q`: axis ratio
    """
    return I0 * np.pi * rd**2 * q / (n - 1)
