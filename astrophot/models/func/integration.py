from typing import Tuple
import numpy as np

from ...utils.integration import quad_table
from ...backend_obj import backend, ArrayLike
from ... import config


def pixel_center_integrator(Z: ArrayLike) -> ArrayLike:
    return Z


def pixel_simpsons_integrator(Z: ArrayLike) -> ArrayLike:
    kernel = (
        backend.as_array(
            [[[[1, 4, 1], [4, 16, 4], [1, 4, 1]]]], dtype=config.DTYPE, device=config.DEVICE
        )
        / 36.0
    )
    Z = backend.conv2d(Z.reshape(1, 1, *Z.shape), kernel, padding="valid", stride=2)
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
        _, _, w = quad_table(order, config.DTYPE, config.DEVICE)
    Z = Z * w
    return backend.sum(Z, dim=-1)


def upsample(i: ArrayLike, j: ArrayLike, order: int, scale: float) -> Tuple[ArrayLike, ArrayLike]:
    dp = (
        backend.linspace(-1, 1, order, dtype=config.DTYPE, device=config.DEVICE)
        * (order - 1)
        / (2.0 * order)
    )
    di, dj = backend.meshgrid(dp, dp, indexing="xy")

    si = backend.repeat(i[..., None], order**2, -1) + scale * di.flatten()
    sj = backend.repeat(j[..., None], order**2, -1) + scale * dj.flatten()
    return si, sj


def single_quad_integrate(
    i: ArrayLike, j: ArrayLike, brightness_ij, scale: float, quad_order: int = 3
) -> Tuple[ArrayLike, ArrayLike]:
    di, dj, w = quad_table(quad_order, config.DTYPE, config.DEVICE)
    qi = backend.repeat(i[..., None], quad_order**2, -1) + scale * di.flatten()
    qj = backend.repeat(j[..., None], quad_order**2, -1) + scale * dj.flatten()
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
    select = backend.topk(backend.abs(z - z0).flatten(), N)[1]

    integral_flat = z.flatten()

    si, sj = upsample(i.flatten()[select], j.flatten()[select], gridding, scale)

    integral_flat = backend.fill_at_indices(
        integral_flat,
        select,
        backend.mean(
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
        ),
    )

    return integral_flat.reshape(z.shape)


def bright_integrate(
    z: ArrayLike,
    i: ArrayLike,
    j: ArrayLike,
    brightness_ij: callable,
    bright_frac: float,
    scale: float = 1.0,
    quad_order: int = 3,
    gridding: int = 5,
    max_depth: int = 2,
):
    trace = []
    for d in range(max_depth):
        N = max(1, int(np.prod(z.shape) * bright_frac))
        z_flat = z.flatten()
        select = backend.topk(z_flat, N)[1]
        trace.append([z_flat, select, z.shape])
        if d > 0:
            i, j = upsample(i.flatten()[select], j.flatten()[select], gridding, scale)
            scale = scale / gridding
        else:
            i, j = i.flatten()[select].reshape(-1, 1), j.flatten()[select].reshape(-1, 1)
        z, _ = single_quad_integrate(i, j, brightness_ij, scale, quad_order)
    trace.append([z, None, z.shape])

    for _ in reversed(range(1, max_depth + 1)):
        T = trace.pop(-1)
        trace[-1][0] = backend.fill_at_indices(
            trace[-1][0], trace[-1][1], backend.mean(T[0].reshape(T[2]), dim=-1)
        )

    return trace[0][0].reshape(trace[0][2])
