import numpy as np
import torch

deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi
rad_to_arcsec = rad_to_deg * 3600
arcsec_to_rad = deg_to_rad / 3600


def world_to_plane_gnomonic(ra, dec, ra0, dec0, x0=0.0, y0=0.0):
    """
    Convert world coordinates (RA, Dec) to plane coordinates (x, y) using the gnomonic projection.

    Parameters
    ----------
    ra : torch.Tensor
        Right Ascension in degrees.
    dec : torch.Tensor
        Declination in degrees.
    ra0 : torch.Tensor
        Reference Right Ascension in degrees.
    dec0 : torch.Tensor
        Reference Declination in degrees.

    Returns
    -------
    x : torch.Tensor
        x coordinate in arcseconds.
    y : torch.Tensor
        y coordinate in arcseconds.
    """
    ra = ra * deg_to_rad
    dec = dec * deg_to_rad
    ra0 = ra0 * deg_to_rad
    dec0 = dec0 * deg_to_rad

    cosc = torch.sin(dec0) * torch.sin(dec) + torch.cos(dec0) * torch.cos(dec) * torch.cos(ra - ra0)

    x = torch.cos(dec) * torch.sin(ra - ra0)

    y = torch.cos(dec0) * torch.sin(dec) - torch.sin(dec0) * torch.cos(dec) * torch.cos(ra - ra0)

    return x * rad_to_arcsec / cosc + x0, y * rad_to_arcsec / cosc + y0


def plane_to_world_gnomonic(x, y, ra0, dec0, x0=0.0, y0=0.0, s=1e-3):
    """
    Convert plane coordinates (x, y) to world coordinates (RA, Dec) using the gnomonic projection.
    Parameters
    ----------
    x : torch.Tensor
        x coordinate in arcseconds.
    y : torch.Tensor
        y coordinate in arcseconds.
    ra0 : torch.Tensor
        Reference Right Ascension in degrees.
    dec0 : torch.Tensor
        Reference Declination in degrees.
    s : float
        Small constant to avoid division by zero.

    Returns
    -------
    ra : torch.Tensor
        Right Ascension in degrees.
    dec : torch.Tensor
        Declination in degrees.
    """
    x = (x - x0) * arcsec_to_rad
    y = (y - y0) * arcsec_to_rad
    ra0 = ra0 * deg_to_rad
    dec0 = dec0 * deg_to_rad

    rho = torch.sqrt(x**2 + y**2) + s
    c = torch.arctan(rho)

    ra = ra0 + torch.arctan2(
        x * torch.sin(c),
        rho * torch.cos(dec0) * torch.cos(c) - y * torch.sin(dec0) * torch.sin(c),
    )

    dec = torch.arcsin(torch.cos(c) * torch.sin(dec0) + y * torch.sin(c) * torch.cos(dec0) / rho)

    return ra * rad_to_deg, dec * rad_to_deg


def pixel_to_plane_linear(i, j, i0, j0, CD, x0=0.0, y0=0.0):
    """
    Convert pixel coordinates to a tangent plane using the WCS information. This
    matches the FITS convention for linear transformations.

    Parameters
    ----------
    i: Tensor
        The first coordinate of the pixel in pixel units.
    j: Tensor
        The second coordinate of the pixel in pixel units.
    i0: Tensor
        The i reference pixel coordinate in pixel units.
    j0: Tensor
        The j reference pixel coordinate in pixel units.
    CD: Tensor
        The CD matrix in arcsec per pixel. This 2x2 matrix is used to convert
        from pixel to arcsec units and also handles rotation/skew.
    x0: float
        The x reference coordinate in arcsec.
    y0: float
        The y reference coordinate in arcsec.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        Tuple containing the x and y tangent plane coordinates in arcsec.
    """
    uv = torch.stack((j.reshape(-1) - j0, i.reshape(-1) - i0), dim=1)
    xy = (CD @ uv.T).T

    return xy[:, 0].reshape(i.shape) + x0, xy[:, 1].reshape(j.shape) + y0


def sip_delta(u, v, sipA=(), sipB=()):
    """
    u = j - j0
    v = i - i0
    sipA = dict(tuple(int,int), float)
        The SIP coefficients, where the keys are tuples of powers (i, j) and the values are the coefficients.
        For example, {(1, 2): 0.1} means delta_u = 0.1 * (u * v^2).
    """
    delta_u = torch.zeros_like(u)
    delta_v = torch.zeros_like(v)
    # Get all used coefficient powers
    all_a = set(s[0] for s in sipA) | set(s[0] for s in sipB)
    all_b = set(s[1] for s in sipA) | set(s[1] for s in sipB)
    # Pre-compute all powers of u and v
    u_a = dict((a, u**a) for a in all_a)
    v_b = dict((b, v**b) for b in all_b)
    for a, b in sipA:
        delta_u = delta_u + sipA[(a, b)] * (u_a[a] * v_b[b])
    for a, b in sipB:
        delta_v = delta_v + sipB[(a, b)] * (u_a[a] * v_b[b])
    return delta_u, delta_v


def pixel_to_plane_sip(i, j, i0, j0, CD, sip_powers=[], sip_coefs=[], x0=0.0, y0=0.0):
    """
    Convert pixel coordinates to a tangent plane using the WCS information. This
    matches the FITS convention for SIP transformations.

    For more information see:

    * FITS World Coordinate System (WCS):
      https://fits.gsfc.nasa.gov/fits_wcs.html
    * Representations of world coordinates in FITS, 2002, by Geisen and
      Calabretta
    * The SIP Convention for Representing Distortion in FITS Image Headers,
      2008, by Shupe and Hook

    Parameters
    ----------
    i: Tensor
        The first coordinate of the pixel in pixel units.
    j: Tensor
        The second coordinate of the pixel in pixel units.
    i0: Tensor
        The i reference pixel coordinate in pixel units.
    j0: Tensor
        The j reference pixel coordinate in pixel units.
    CD: Tensor
        The CD matrix in degrees per pixel. This 2x2 matrix is used to convert
        from pixel to degree units and also handles rotation/skew.
    sip_powers: Tensor
        The powers of the pixel coordinates for the SIP distortion, should be a
        shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the powers in order ``i,
        j``.
    sip_coefs: Tensor
        The coefficients of the pixel coordinates for the SIP distortion, should
        be a shape (N orders, 2) tensor. ``N orders`` is the number of non-zero
        polynomial coefficients. The second axis has the coefficients in order
        ``delta_x, delta_y``.
    x0: float
        The x reference coordinate in arcsec.
    y0: float
        The y reference coordinate in arcsec.

    Note
    ----
    The representation of the SIP powers and coefficients assumes that the SIP
    polynomial will use the same orders for both the x and y coordinates. If
    this is not the case you may use zeros for the coefficients to ensure all
    polynomial combinations are evaluated. However, it is very common to have
    the same orders for both.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        Tuple containing the x and y tangent plane coordinates in arcsec.
    """
    uv = torch.stack((j.reshape(-1) - j0, i.reshape(-1) - i0), dim=1)
    delta_p = torch.zeros_like(uv)
    for p in range(len(sip_powers)):
        delta_p += sip_coefs[p] * torch.prod(uv ** sip_powers[p], dim=-1).unsqueeze(-1)
    plane = torch.einsum("ij,...j->...i", CD, uv + delta_p)
    return plane[..., 0] + x0, plane[..., 1] + y0


def plane_to_pixel_linear(x, y, i0, j0, iCD, x0=0.0, y0=0.0):
    """
    Convert tangent plane coordinates to pixel coordinates using the WCS
    information. This matches the FITS convention for linear transformations.

    Parameters
    ----------
    x: Tensor
        The first coordinate of the pixel in arcsec.
    y: Tensor
        The second coordinate of the pixel in arcsec.
    i0: Tensor
        The i reference pixel coordinate in pixel units.
    j0: Tensor
        The j reference pixel coordinate in pixel units.
    iCD: Tensor
        The inverse CD matrix in arcsec per pixel. This 2x2 matrix is used to convert
        from pixel to arcsec units and also handles rotation/skew.
    x0: float
        The x reference coordinate in arcsec.
    y0: float
        The y reference coordinate in arcsec.

    Returns
    -------
    Tuple: [Tensor, Tensor]
        Tuple containing the i and j pixel coordinates in pixel units.
    """
    xy = torch.stack((x.reshape(-1) - x0, y.reshape(-1) - y0), dim=1)
    uv = (iCD @ xy.T).T

    return uv[:, 1].reshape(x.shape) + i0, uv[:, 0].reshape(y.shape) + j0
