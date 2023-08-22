import numpy as np

deg_to_arcsec = 3600.

def flux_to_sb(flux, pixel_area, zeropoint):
    """Conversion from flux units to logarithmic surface brightness
    units.

    .. math::

      \\mu = -2.5\\log_{10}(flux) + z.p. + 2.5\\log_{10}(A)

    where :math:`z.p.` is the zeropoint and :math:`A` is the area of a pixel.

    """
    return -2.5 * np.log10(flux) + zeropoint + 2.5 * np.log10(pixel_area)


def flux_to_mag(flux, zeropoint, fluxe=None):
    """Converts a flux total into logarithmic magnitude units.

    .. math::

      m = -2.5\\log_{10}(flux) + z.p.

    where :math:`z.p.` is the zeropoint.
    
    """
    if fluxe is None:
        return -2.5 * np.log10(flux) + zeropoint
    else:
        return -2.5 * np.log10(flux) + zeropoint, 2.5 * fluxe / (np.log(10) * flux)


def sb_to_flux(sb, pixel_area, zeropoint):
    """Converts logarithmic surface brightness units into flux units.

    .. math::

      flux = A 10^{-(\\mu - z.p.)/2.5}

    where :math:`z.p.` is the zeropoint and :math:`A` is the area of a pixel.
    
    """
    return pixel_area * 10 ** (-(sb - zeropoint) / 2.5)


def mag_to_flux(mag, zeropoint, mage=None):
    """converts logarithmic magnitude units into a flux total.

    .. math::

      flux = 10^{-(m - z.p.)/2.5}

    where :math:`z.p.` is the zeropoint.

    """
    if mage is None:
        return 10 ** (-(mag - zeropoint) / 2.5)
    else:
        I = 10 ** (-(mag - zeropoint) / 2.5)
        return I, np.log(10) * I * mage / 2.5


def magperarcsec2_to_mag(mu, a=None, b=None, A=None):
    """
    Converts mag/arcsec^2 to mag
    mu: mag/arcsec^2
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag


    .. math::

      m = \\mu -2.5\\log_{10}(A)

    where :math:`A` is an area in arcsec^2.
    
    """
    assert (not A is None) or (not a is None and not b is None)
    if A is None:
        A = np.pi * a * b
    return mu - 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness


def mag_to_magperarcsec2(m, a=None, b=None, R=None, A=None):
    """
    Converts mag to mag/arcsec^2
    m: mag
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag/arcsec^2

    .. math::

      \\mu = m + 2.5\\log_{10}(A)

    where :math:`A` is an area in arcsec^2.    
    """
    assert (not A is None) or (not a is None and not b is None) or (not R is None)
    if not R is None:
        A = np.pi * (R ** 2)
    elif A is None:
        A = np.pi * a * b
    return m + 2.5 * np.log10(
        A
    )  # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness


def PA_shift_convention(pa, unit="rad"):
    """
    Alternates between standard mathematical convention for angles, and astronomical position angle convention.
    The standard convention is to measure angles counter-clockwise relative to the positive x-axis
    The astronomical convention is to measure angles counter-clockwise relative to the positive y-axis
    """

    if unit == "rad":
        shift = np.pi
    elif unit == "deg":
        shift = 180.0

    return (pa - (shift / 2)) % shift
