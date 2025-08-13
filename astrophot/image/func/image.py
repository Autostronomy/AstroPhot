from ...utils.integration import quad_table
from ...backend_obj import backend, ArrayLike


def pixel_center_meshgrid(shape: tuple[int, int], dtype, device) -> tuple:
    i = backend.arange(shape[0], dtype=dtype, device=device)
    j = backend.arange(shape[1], dtype=dtype, device=device)
    return backend.meshgrid(i, j, indexing="ij")


def cmos_pixel_center_meshgrid(
    shape: tuple[int, int], loc: tuple[float, float], dtype, device
) -> tuple:
    i = backend.arange(shape[0], dtype=dtype, device=device) + loc[0]
    j = backend.arange(shape[1], dtype=dtype, device=device) + loc[1]
    return backend.meshgrid(i, j, indexing="ij")


def pixel_corner_meshgrid(shape: tuple[int, int], dtype, device) -> tuple:
    i = backend.arange(shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = backend.arange(shape[1] + 1, dtype=dtype, device=device) - 0.5
    return backend.meshgrid(i, j, indexing="ij")


def pixel_simpsons_meshgrid(shape: tuple[int, int], dtype, device) -> tuple:
    i = 0.5 * backend.arange(2 * shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = 0.5 * backend.arange(2 * shape[1] + 1, dtype=dtype, device=device) - 0.5
    return backend.meshgrid(i, j, indexing="ij")


def pixel_quad_meshgrid(shape: tuple[int, int], dtype, device, order=3) -> tuple:
    i, j = pixel_center_meshgrid(shape, dtype, device)
    di, dj, w = quad_table(order, dtype, device)
    i = backend.repeat(i[..., None], order**2, -1) + di.flatten()
    j = backend.repeat(j[..., None], order**2, -1) + dj.flatten()
    return i, j, w.flatten()


def rotate(theta: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple:
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y
