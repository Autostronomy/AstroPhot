from typing import Tuple
import torch
import numpy as np

from ...utils.integration import quad_table
from ...backend_obj import backend, ArrayLike


def pixel_center_integrator(Z: ArrayLike) -> ArrayLike:
    return Z


def pixel_corner_integrator(Z: ArrayLike) -> ArrayLike:
    kernel = backend.ones((1, 1, 2, 2), dtype=Z.dtype, device=Z.device) / 4.0
    Z = backend.conv2d(Z.view(1, 1, *Z.shape), kernel, padding="valid")
    return Z.squeeze(0).squeeze(0)


def pixel_simpsons_integrator(Z: ArrayLike) -> ArrayLike:
    kernel = (
        backend.as_array([[[[1, 4, 1], [4, 16, 4], [1, 4, 1]]]], dtype=Z.dtype, device=Z.device)
        / 36.0
    )
    Z = backend.conv2d(Z.view(1, 1, *Z.shape), kernel, padding="valid", stride=2)
    return Z.squeeze(0).squeeze(0)


def pixel_quad_integrator(Z: ArrayLike, w: ArrayLike = None, order: int = 3) -> ArrayLike:
    """
    Integrate the pixel values using quadrature weights.

    **Args:**
    -  `Z`: The tensor containing pixel values.
    -  `w`: The quadrature weights.
    -  `order`: The order of the quadrature.
    """
    if w is None:
        _, _, w = quad_table(order, Z.dtype, Z.device)
    Z = Z * w
    return Z.sum(dim=(-1))


def upsample(i: ArrayLike, j: ArrayLike, order: int, scale: float) -> Tuple[ArrayLike, ArrayLike]:
    dp = (
        backend.linspace(-1, 1, order, dtype=i.dtype, device=i.device) * (order - 1) / (2.0 * order)
    )
    di, dj = backend.meshgrid(dp, dp, indexing="xy")

    si = backend.repeat(i.unsqueeze(-1), order**2, -1) + scale * di.flatten()
    sj = backend.repeat(j.unsqueeze(-1), order**2, -1) + scale * dj.flatten()
    return si, sj


def single_quad_integrate(
    i: ArrayLike, j: ArrayLike, brightness_ij, scale: float, quad_order: int = 3
) -> Tuple[ArrayLike, ArrayLike]:
    di, dj, w = quad_table(quad_order, i.dtype, i.device)
    qi = backend.repeat(i.unsqueeze(-1), quad_order**2, -1) + scale * di.flatten()
    qj = backend.repeat(j.unsqueeze(-1), quad_order**2, -1) + scale * dj.flatten()
    z = brightness_ij(qi, qj)
    z0 = backend.mean(z, dim=-1)
    z = backend.sum(z * w.flatten(), dim=-1)
    return z, z0


def recursive_quad_integrate(
    i: ArrayLike,
    j: ArrayLike,
    brightness_ij: callable,
    curve_frac: float,
    scale: float = 1.0,
    quad_order: int = 3,
    gridding: int = 5,
    _current_depth: int = 0,
    max_depth: int = 1,
) -> ArrayLike:
    z, z0 = single_quad_integrate(i, j, brightness_ij, scale, quad_order)

    if _current_depth >= max_depth:
        return z

    N = max(1, int(np.prod(z.shape) * curve_frac))
    select = backend.topk(backend.abs(z - z0).flatten(), N, dim=-1).indices

    integral_flat = z.flatten()

    si, sj = upsample(i.flatten()[select], j.flatten()[select], quad_order, scale)

    integral_flat[select] = backend.mean(
        recursive_quad_integrate(
            si,
            sj,
            brightness_ij,
            curve_frac=curve_frac,
            scale=scale / gridding,
            quad_order=quad_order,
            gridding=gridding,
            _current_depth=_current_depth + 1,
            max_depth=max_depth,
        ),
        dim=-1,
    )

    return integral_flat.reshape(z.shape)


def recursive_bright_integrate(
    i: ArrayLike,
    j: ArrayLike,
    brightness_ij: callable,
    bright_frac: float,
    scale: float = 1.0,
    quad_order: int = 3,
    gridding: int = 5,
    _current_depth: int = 0,
    max_depth: int = 1,
) -> ArrayLike:
    z, _ = single_quad_integrate(i, j, brightness_ij, scale, quad_order)

    if _current_depth >= max_depth:
        return z

    N = max(1, int(np.prod(z.shape) * bright_frac))
    z_flat = z.flatten()

    select = backend.topk(z_flat, N, dim=-1).indices

    si, sj = upsample(i.flatten()[select], j.flatten()[select], quad_order, scale)

    z_flat[select] = backend.mean(
        recursive_bright_integrate(
            si,
            sj,
            brightness_ij,
            bright_frac,
            scale=scale / gridding,
            quad_order=quad_order,
            gridding=gridding,
            _current_depth=_current_depth + 1,
            max_depth=max_depth,
        ),
        dim=-1,
    )

    return z_flat.reshape(z.shape)
