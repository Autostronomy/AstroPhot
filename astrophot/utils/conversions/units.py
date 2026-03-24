from typing import Optional
import numpy as np

__all__ = (
    "deg_to_arcsec",
    "arcsec_to_deg",
    "flux_to_sb",
    "flux_to_mag",
    "sb_to_flux",
    "mag_to_flux",
    "magperarcsec2_to_mag",
    "mag_to_magperarcsec2",
    "PA_shift_convention",
)

deg_to_arcsec = 3600.0
arcsec_to_deg = 1.0 / deg_to_arcsec


def flux_to_sb(flux: float, pixel_area: float, zeropoint: float) -> float:
    """Conversion from flux units to logarithmic surface brightness
    units.

    $$\\mu = -2.5\\log_{10}(flux) + z.p. + 2.5\\log_{10}(A)$$

    where $z.p.$ is the zeropoint and $A$ is the area of a pixel.

    """
    return -2.5 * np.log10(flux) + zeropoint + 2.5 * np.log10(pixel_area)


def flux_to_mag(flux: float, zeropoint: float, fluxe: Optional[float] = None) -> float:
    """Converts a flux total into logarithmic magnitude units.

    $$m = -2.5\\log_{10}(flux) + z.p.$$

    where $z.p.$ is the zeropoint.

    """
    if fluxe is None:
        return -2.5 * np.log10(flux) + zeropoint
    else:
        return -2.5 * np.log10(flux) + zeropoint, 2.5 * fluxe / (np.log(10) * flux)


def sb_to_flux(sb: float, pixel_area: float, zeropoint: float) -> float:
    """Converts logarithmic surface brightness units into flux units.

    $$flux = A 10^{-(\\mu - z.p.)/2.5}$$

    where $z.p.$ is the zeropoint and $A$ is the area of a pixel.

    """
    return pixel_area * 10 ** (-(sb - zeropoint) / 2.5)


def mag_to_flux(mag: float, zeropoint: float, mage: Optional[float] = None) -> float:
    """converts logarithmic magnitude units into a flux total.

    $$flux = 10^{-(m - z.p.)/2.5}$$

    where $z.p.$ is the zeropoint.

    """
    if mage is None:
        return 10 ** (-(mag - zeropoint) / 2.5)
    else:
        I = 10 ** (-(mag - zeropoint) / 2.5)
        return I, np.log(10) * I * mage / 2.5


def magperarcsec2_to_mag(
    mu: float, a: Optional[float] = None, b: Optional[float] = None, A: Optional[float] = None
) -> float:
    """
    Converts mag/arcsec^2 to mag

    **Args:**
    - `mu`: mag/arcsec^2
    - `a`: semi major axis radius (arcsec)
    - `b`: semi minor axis radius (arcsec)
    - `A`: pre-calculated area (arcsec^2)


    $$m = \\mu -2.5\\log_{10}(A)$$

    where $A$ is an area in arcsec^2.

    """
    assert (A is not None) or (a is not None and b is not None)
    if A is None:
        A = np.pi * a * b
    return mu - 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness


def mag_to_magperarcsec2(
    m: float,
    a: Optional[float] = None,
    b: Optional[float] = None,
    R: Optional[float] = None,
    A: Optional[float] = None,
) -> float:
    """
    Converts mag to mag/arcsec^2

    **Args:**
    - `m`: mag
    - `a`: semi major axis radius (arcsec)
    - `b`: semi minor axis radius (arcsec)
    - `A`: pre-calculated area (arcsec^2)


    $$\\mu = m + 2.5\\log_{10}(A)$$

    where $A$ is an area in arcsec^2.
    """
    assert (A is not None) or (a is not None and b is not None) or (R is not None)
    if R is not None:
        A = np.pi * (R**2)
    elif A is None:
        A = np.pi * a * b
    return m + 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness
