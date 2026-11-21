import numpy as np
from ...backend_obj import backend
from ... import config

deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi
rad_to_arcsec = rad_to_deg * 3600
arcsec_to_rad = deg_to_rad / 3600


def world_to_plane_gnomonic(ra, dec, ra0, dec0, x0=0.0, y0=0.0):
    """
    Convert world coordinates (RA, Dec) to plane coordinates (x, y) using the gnomonic projection.

    :param ra: Right Ascension in degrees.
    :type ra: Array
    :param dec: Declination in degrees.
    :type dec: Array
    :param ra0: Reference Right Ascension in degrees.
    :type ra0: Array
    :param dec0: Reference Declination in degrees.
    :type dec0: Array
    :returns: Tuple containing ``x`` and ``y`` coordinates in arcseconds.
    :rtype: tuple[Array, Array]
    """
    ra = ra * deg_to_rad
    dec = dec * deg_to_rad
    ra0 = ra0 * deg_to_rad
    dec0 = dec0 * deg_to_rad

    cosc = backend.sin(dec0) * backend.sin(dec) + backend.cos(dec0) * backend.cos(
        dec
    ) * backend.cos(ra - ra0)

    x = backend.cos(dec) * backend.sin(ra - ra0)

    y = backend.cos(dec0) * backend.sin(dec) - backend.sin(dec0) * backend.cos(dec) * backend.cos(
        ra - ra0
    )

    return x * rad_to_arcsec / cosc + x0, y * rad_to_arcsec / cosc + y0


def plane_to_world_gnomonic(x, y, ra0, dec0, x0=0.0, y0=0.0, s=1e-10):
    """
    Convert plane coordinates (x, y) to world coordinates (RA, Dec) using the gnomonic projection.

    :param x: x coordinate in arcseconds.
    :type x: Array
    :param y: y coordinate in arcseconds.
    :type y: Array
    :param ra0: Reference Right Ascension in degrees.
    :type ra0: Array
    :param dec0: Reference Declination in degrees.
    :type dec0: Array
    :param s: Small constant to avoid division by zero.
    :type s: float
    :returns: Tuple containing ``ra`` and ``dec`` in degrees.
    :rtype: tuple[Array, Array]
    """
    x = (x - x0) * arcsec_to_rad
    y = (y - y0) * arcsec_to_rad
    ra0 = ra0 * deg_to_rad
    dec0 = dec0 * deg_to_rad

    rho = backend.sqrt(x**2 + y**2) + s
    c = backend.arctan(rho)

    ra = ra0 + backend.arctan2(
        x * backend.sin(c),
        rho * backend.cos(dec0) * backend.cos(c) - y * backend.sin(dec0) * backend.sin(c),
    )

    dec = backend.arcsin(
        backend.cos(c) * backend.sin(dec0) + y * backend.sin(c) * backend.cos(dec0) / rho
    )

    return ra * rad_to_deg, dec * rad_to_deg


def tangent_plane_wcs(wcs, shape=None, step=10.0):
    """
    Tangent plane (gnomonic) approximation of an astropy WCS which has another
    projection, or another celestial frame than RA and Dec.

    The tangent point is put at the centre of the image and the CD matrix is
    found by finite differences of the WCS about it. The approximation is exact
    at the centre and to first order around it. The sky coordinates are
    converted to ICRS, so galactic or ecliptic axes can be used as well.

    :param wcs: The WCS to approximate.
    :type wcs: astropy.wcs.WCS
    :param shape: Number of pixels on FITS axis 1 and axis 2. Without it the
      tangent point is the reference pixel of the WCS.
    :type shape: tuple[int, int]
    :param step: Number of pixels to each side of the tangent point that the
      finite differences are taken over.
    :type step: float
    :returns: ``crval`` in degrees, ``crpix`` as a 0-indexed pixel, ``CD`` in
      arcsec per pixel and ``max_offset``, the largest distance in arcsec
      between the approximation and the WCS at the corners of the image (None
      without ``shape``).
    :rtype: dict
    :raises ValueError: If astropy can not convert the coordinates of the WCS to ICRS.
    """
    wcs = wcs.celestial
    if shape is None or min(shape) == 0:
        crpix = np.array(wcs.wcs.crpix, dtype=np.float64) - 1
    else:
        crpix = (np.array(shape, dtype=np.float64) - 1) / 2

    def sky(i, j):
        coord = wcs.pixel_to_world(i, j)
        if not hasattr(coord, "icrs"):
            raise ValueError(f"astropy has no celestial frame for CTYPE {tuple(wcs.wcs.ctype)}")
        return coord.icrs.ra.rad, coord.icrs.dec.rad

    ra0, dec0 = sky(*crpix)

    def plane(i, j):
        # the gnomonic projection about the tangent point, in arcsec
        ra, dec = sky(i, j)
        cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        x = np.cos(dec) * np.sin(ra - ra0)
        y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
        return np.stack((x, y)) * rad_to_arcsec / cosc

    # Projecting the stepped points onto the new tangent plane gives the matrix
    # in the coordinates that AstroPhot works in, which holds at any Dec
    step_i = plane(crpix[0] + step * np.array([-1.0, 1.0]), crpix[1] + np.zeros(2))
    step_j = plane(crpix[0] + np.zeros(2), crpix[1] + step * np.array([-1.0, 1.0]))
    CD = np.stack((step_i[:, 1] - step_i[:, 0], step_j[:, 1] - step_j[:, 0]), axis=1) / (2 * step)

    max_offset = None
    if shape is not None and min(shape) > 0:
        corners = np.array(
            [[-0.5, shape[0] - 0.5, -0.5, shape[0] - 0.5], [-0.5] * 2 + [shape[1] - 0.5] * 2]
        )
        offset = CD @ (corners - crpix[:, None]) - plane(*corners)
        max_offset = float(np.hypot(*offset).max())

    return {
        "crval": np.array([ra0, dec0]) * rad_to_deg,
        "crpix": crpix,
        "CD": CD,
        "max_offset": max_offset,
    }


def pixel_to_plane_linear(i, j, i0, j0, CD, x0=0.0, y0=0.0):
    """
    Convert pixel coordinates to a tangent plane using the WCS information. This
    matches the FITS convention for linear transformations.

    :param i: The first coordinate of the pixel in pixel units.
    :type i: Array
    :param j: The second coordinate of the pixel in pixel units.
    :type j: Array
    :param i0: The i reference pixel coordinate in pixel units.
    :type i0: Array
    :param j0: The j reference pixel coordinate in pixel units.
    :type j0: Array
    :param CD: The CD matrix in arcsec per pixel. This 2x2 matrix converts
      from pixel to arcsec units and also handles rotation/skew.
    :type CD: Array
    :param x0: The x reference coordinate in arcseconds.
    :type x0: float
    :param y0: The y reference coordinate in arcseconds.
    :type y0: float

    :returns: Tuple containing the x and y coordinates in arcseconds
    :rtype: tuple[Array, Array]
    """
    uv = backend.stack((i.flatten() - i0, j.flatten() - j0), dim=0)
    xy = CD @ uv

    return xy[0].reshape(i.shape) + x0, xy[1].reshape(i.shape) + y0


def sip_coefs(order):
    coefs = []
    for p in range(order + 1):
        for q in range(order + 1 - p):
            coefs.append((p, q))
    return tuple(coefs)


def sip_matrix(u, v, order):
    M = backend.zeros(
        (len(u), (order + 1) * (order + 2) // 2), dtype=config.DTYPE, device=config.DEVICE
    )
    for i, (p, q) in enumerate(sip_coefs(order)):
        M = backend.fill_at_indices(M, (slice(None), i), u**p * v**q)
    return M


def sip_backward_transform(u, v, U, V, A_ORDER, B_ORDER):
    """
    Credit: Shu Liu and Lei Hi, see here:
    https://github.com/Roman-Supernova-PIT/sfft/blob/master/sfft/utils/CupyWCSTransform.py

    Compute the backward transformation from (U, V) to (u, v)
    """

    FP_UV = sip_matrix(U, V, A_ORDER)
    GP_UV = sip_matrix(U, V, B_ORDER)

    AP = backend.linalg.lstsq(FP_UV, (u.flatten() - U).reshape(-1, 1))[0].squeeze(1)
    BP = backend.linalg.lstsq(GP_UV, (v.flatten() - V).reshape(-1, 1))[0].squeeze(1)
    return AP, BP


def sip_delta(u, v, sipA=(), sipB=()):
    """
    u = i - i0
    v = j - j0
    sipA = dict(tuple(int,int), float)
        The SIP coefficients, where the keys are tuples of powers (i, j) and the values are the coefficients.
        For example, {(1, 2): 0.1} means delta_u = 0.1 * (u * v^2).
    """
    delta_u = backend.zeros_like(u)
    delta_v = backend.zeros_like(v)
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


def plane_to_pixel_linear(x, y, i0, j0, CD, x0=0.0, y0=0.0):
    """
    Convert tangent plane coordinates to pixel coordinates using the WCS
    information. This matches the FITS convention for linear transformations.

    :param x: The first coordinate of the pixel in arcsec.
    :type x: Array
    :param y: The second coordinate of the pixel in arcsec.
    :type y: Array
    :param i0: The i reference pixel coordinate in pixel units.
    :type i0: Array
    :param j0: The j reference pixel coordinate in pixel units.
    :type j0: Array
    :param CD: The CD matrix in arcsec per pixel.
    :type CD: Array
    :param x0: The x reference coordinate in arcsec.
    :type x0: float
    :param y0: The y reference coordinate in arcsec.
    :type y0: float

    :returns: Tuple containing the i and j pixel coordinates in pixel units.
    :rtype: tuple[Array, Array]
    """
    xy = backend.stack((x.flatten() - x0, y.flatten() - y0), dim=0)
    uv = backend.linalg.inv(CD) @ xy

    return uv[0].reshape(x.shape) + i0, uv[1].reshape(y.shape) + j0
