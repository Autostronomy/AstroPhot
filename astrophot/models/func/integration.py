import torch
from functools import lru_cache

from scipy.special import roots_legendre


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
    di, dj = torch.meshgrid(a, a, indexing="xy")

    w = torch.outer(w, w) / 4.0
    return di, dj, w


def pixel_center_meshgrid(shape, dtype, device):
    i = torch.arange(shape[0], dtype=dtype, device=device)
    j = torch.arange(shape[1], dtype=dtype, device=device)
    return torch.meshgrid(i, j, indexing="xy")


def pixel_center_integrator(Z: torch.Tensor):
    return Z


def pixel_corner_meshgrid(shape, dtype, device):
    i = torch.arange(shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = torch.arange(shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="xy")


def pixel_corner_integrator(Z: torch.Tensor):
    kernel = torch.ones((1, 1, 2, 2), dtype=Z.dtype, device=Z.device) / 4.0
    Z = torch.nn.functional.conv2d(Z.view(1, 1, *Z.shape), kernel, padding="valid")
    return Z.squeeze(0).squeeze(0)


def pixel_simpsons_meshgrid(shape, dtype, device):
    i = 0.5 * torch.arange(2 * shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = 0.5 * torch.arange(2 * shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="xy")


def pixel_simpsons_integrator(Z: torch.Tensor):
    kernel = (
        torch.tensor([[[[1, 4, 1], [4, 16, 4], [1, 4, 1]]]], dtype=Z.dtype, device=Z.device) / 36.0
    )
    Z = torch.nn.functional.conv2d(Z.view(1, 1, *Z.shape), kernel, padding="valid", stride=2)
    return Z.squeeze(0).squeeze(0)


def pixel_quad_meshgrid(shape, dtype, device, order=3):
    i, j = pixel_center_meshgrid(shape, dtype, device)
    di, dj, w = quad_table(order, dtype, device)
    i = torch.repeat_interleave(i[..., None], order**2, -1) + di
    j = torch.repeat_interleave(j[..., None], order**2, -1) + dj
    return i, j, w


def pixel_quad_integrator(Z: torch.Tensor, w: torch.Tensor = None, order=3):
    """
    Integrate the pixel values using quadrature weights.

    Parameters
    ----------
    Z : torch.Tensor
        The tensor containing pixel values.
    w : torch.Tensor
        The quadrature weights.

    Returns
    -------
    torch.Tensor
        The integrated value.
    """
    if w is None:
        _, _, w = _quad_table(order, Z.dtype, Z.device)
    Z = Z * w
    return Z.sum(dim=(-2, -1))
