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

    **Args:**
    - `ra`: (torch.Tensor) Right Ascension in degrees.
    - `dec`: (torch.Tensor) Declination in degrees.
    - `ra0`: (torch.Tensor) Reference Right Ascension in degrees.
    - `dec0`: (torch.Tensor) Reference Declination in degrees.

    **Returns:**
    - `x`: (torch.Tensor) x coordinate in arcseconds.
    - `y`: (torch.Tensor) y coordinate in arcseconds.
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

    **Args:**
    - `x`: (Tensor) x coordinate in arcseconds.
    - `y`: (Tensor) y coordinate in arcseconds.
    - `ra0`: (Tensor) Reference Right Ascension in degrees.
    - `dec0`: (Tensor) Reference Declination in degrees.
    - `s`: (float) Small constant to avoid division by zero.

    **Returns:**
    - `ra`: (Tensor) Right Ascension in degrees.
    - `dec`: (Tensor) Declination in degrees.
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


def pixel_to_plane_linear(i, j, i0, j0, CD, x0=0.0, y0=0.0):
    """
    Convert pixel coordinates to a tangent plane using the WCS information. This
    matches the FITS convention for linear transformations.

    **Args:**
    -  `i` (Tensor): The first coordinate of the pixel in pixel units.
    -  `j` (Tensor): The second coordinate of the pixel in pixel units.
    -  `i0` (Tensor): The i reference pixel coordinate in pixel units.
    -  `j0` (Tensor): The j reference pixel coordinate in pixel units.
    -  `CD` (Tensor): The CD matrix in arcsec per pixel. This 2x2 matrix is used to convert
       from pixel to arcsec units and also handles rotation/skew.
    -  `x0` (float): The x reference coordinate in arcseconds.
    -  `y0` (float): The y reference coordinate in arcseconds.

    **Returns:**
    -  Tuple[Tensor, Tensor]: Tuple containing the x and y coordinates in arcseconds
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
    u = j - j0
    v = i - i0
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

    **Args:**
    - `x`: (Tensor) The first coordinate of the pixel in arcsec.
    - `y`: (Tensor) The second coordinate of the pixel in arcsec.
    - `i0`: (Tensor) The i reference pixel coordinate in pixel units.
    - `j0`: (Tensor) The j reference pixel coordinate in pixel units.
    - `CD`: (Tensor) The CD matrix in arcsec per pixel.
    - `x0`: (float) The x reference coordinate in arcsec.
    - `y0`: (float) The y reference coordinate in arcsec.

    **Returns:**
    -  Tuple[Tensor, Tensor]: Tuple containing the i and j pixel coordinates in pixel units.
    """
    xy = backend.stack((x.flatten() - x0, y.flatten() - y0), dim=0)
    uv = backend.linalg.inv(CD) @ xy

    return uv[0].reshape(x.shape) + i0, uv[1].reshape(y.shape) + j0
