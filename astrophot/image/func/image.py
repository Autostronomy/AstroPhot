from ...utils.integration import quad_table
from ...backend_obj import backend, ArrayLike


def pixel_center_meshgrid(
    extent: tuple[int, int, int, int], pad: int, upsample: int, dtype: type, device: str
) -> tuple:
    i = backend.linspace(
        extent[0] - 0.5 + 0.5 / upsample - pad / upsample,
        extent[1] - 0.5 - 0.5 / upsample + pad / upsample,
        (extent[1] - extent[0]) * upsample + 2 * pad,
        dtype=dtype,
        device=device,
    )
    j = backend.linspace(
        extent[2] - 0.5 + 0.5 / upsample - pad / upsample,
        extent[3] - 0.5 - 0.5 / upsample + pad / upsample,
        (extent[3] - extent[2]) * upsample + 2 * pad,
        dtype=dtype,
        device=device,
    )
    return backend.meshgrid(i, j, indexing="ij")


def cmos_pixel_center_meshgrid(
    extent: tuple[int, int, int, int],
    pad: int,
    upsample: int,
    loc: tuple[float, float],
    dtype,
    device,
) -> tuple:
    if upsample > 1:
        raise NotImplementedError("CMOS pixel meshgrid does not currently support upsampling.")
    i, j = pixel_center_meshgrid(extent, pad, upsample, dtype, device)
    return i + loc[0], j + loc[1]


def pixel_corner_meshgrid(
    extent: tuple[int, int, int, int], pad: int, upsample: int, dtype, device
) -> tuple:
    i, j = pixel_center_meshgrid(
        (extent[0], extent[1] + 1, extent[2], extent[3] + 1), pad, upsample, dtype, device
    )
    return i - 0.5 / upsample, j - 0.5 / upsample


def pixel_simpsons_meshgrid(
    extent: tuple[int, int, int, int], pad: int, upsample: int, dtype, device
) -> tuple:
    return pixel_corner_meshgrid(extent, 2 * pad, 2 * upsample, dtype, device)


def pixel_quad_meshgrid(
    extent: tuple[int, int, int, int], pad: int, upsample: int, dtype, device, order=3
) -> tuple:
    i, j = pixel_center_meshgrid(extent, pad, upsample, dtype, device)
    di, dj, w = quad_table(order, dtype, device)
    i = backend.repeat(i[..., None], order**2, -1) + di.flatten() / upsample
    j = backend.repeat(j[..., None], order**2, -1) + dj.flatten() / upsample
    return i, j, w.flatten()


def rotate(theta: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple:
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y
