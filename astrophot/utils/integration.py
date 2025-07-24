from functools import lru_cache

from scipy.special import roots_legendre
import torch


@lru_cache(maxsize=32)
def quad_table(order, dtype, device):
    """
    Generate a meshgrid for quadrature points using Legendre-Gauss quadrature.

    Parameters
    ----------
    n : int
        The number of quadrature points in each dimension.
    dtype : torch.dtype
        The desired data type of the tensor.
    device : torch.device
        The device on which to create the tensor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The generated meshgrid as a tuple of Tensors.
    """
    abscissa, weights = roots_legendre(order)

    w = torch.tensor(weights, dtype=dtype, device=device)
    a = torch.tensor(abscissa, dtype=dtype, device=device) / 2.0
    di, dj = torch.meshgrid(a, a, indexing="ij")

    w = torch.outer(w, w) / 4.0
    return di, dj, w
