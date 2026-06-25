from scipy.special import roots_legendre
from ..backend_obj import backend

__all__ = ("quad_table",)


def quad_table(order, dtype, device):
    """
    Generate a meshgrid for quadrature points using Legendre-Gauss quadrature.

    Parameters
    ----------
    order : int
        The number of quadrature points in each dimension.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, ArrayLike]
        The generated meshgrid as a tuple of Tensors.
    """
    abscissa, weights = roots_legendre(order)

    w = backend.as_array(weights, dtype=dtype, device=device)
    a = backend.as_array(abscissa, dtype=dtype, device=device) / 2.0
    di, dj = backend.meshgrid(a, a, indexing="ij")

    w = backend.outer(w, w) / 4.0
    return di, dj, w
