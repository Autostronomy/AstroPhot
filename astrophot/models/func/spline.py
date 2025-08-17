from ...backend_obj import backend, ArrayLike
from ... import config


def _h_poly(t: ArrayLike) -> ArrayLike:
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Args:
        t (Tensor): A 1D tensor representing the normalized x values.

    Returns:
        Tensor: A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (backend.arange(4, device=config.DEVICE)[:, None])
    A = backend.as_array(
        [
            [1.0, 0.0, -3.0, 2.0],
            [0.0, 1.0, -2.0, 1.0],
            [0.0, 0.0, 3.0, -2.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
        dtype=config.DTYPE,
        device=config.DEVICE,
    )
    return A @ tt


def cubic_spline_torch(x: ArrayLike, y: ArrayLike, xs: ArrayLike) -> ArrayLike:
    """Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    **Args:**
    -  `x` (Tensor): A 1D tensor representing the x-coordinates of the known data points.
    -  `y` (Tensor): A 1D tensor representing the y-coordinates of the known data points.
    -  `xs` (Tensor): A 1D tensor representing the x-coordinates of the positions where
                the cubic spline function should be evaluated.
    """
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = backend.concatenate([m[0].flatten(), (m[1:] + m[:-1]) / 2, m[-1].flatten()])
    idxs = backend.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    return ret


def spline(R: ArrayLike, profR: ArrayLike, profI: ArrayLike, extend: str = "zeros") -> ArrayLike:
    """Spline 1d profile function, cubic spline between points up
    to second last point beyond which is linear

    **Args:**
    -  `R`: Radii tensor at which to evaluate the spline function
    -  `profR`: radius values for the surface density profile in the same units as `R`
    -  `profI`: surface density values for the surface density profile
    -  `extend`: How to extend the spline beyond the last point. Options are 'zeros' or 'const'.
    """
    I = cubic_spline_torch(profR, profI, R.flatten()).reshape(*R.shape)
    if extend == "zeros":
        backend.fill_at_indices(I, R > profR[-1], 0)
    elif extend == "const":
        backend.fill_at_indices(I, R > profR[-1], profI[-1])
    else:
        raise ValueError(f"Unknown extend option: {extend}. Use 'zeros' or 'const'.")
    return I
