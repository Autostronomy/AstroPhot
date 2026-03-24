import torch
import numpy as np

from ..backend_obj import backend, ArrayLike

__all__ = ("default_prof", "interp2d")


def default_prof(
    shape: tuple[int, int], pixelscale: float, min_pixels: int = 2, scale: float = 0.2
) -> np.ndarray:
    prof = [0, min_pixels * pixelscale]
    imagescale = max(shape)  # np.sqrt(np.sum(np.array(shape) ** 2))
    while prof[-1] < (imagescale * pixelscale / 2):
        prof.append(prof[-1] + max(min_pixels * pixelscale, prof[-1] * scale))
    return np.array(prof)


def interp2d(
    im: ArrayLike,
    i: ArrayLike,
    j: ArrayLike,
    padding_mode: str = "zeros",
) -> ArrayLike:
    """
    Interpolates a 2D image at specified coordinates.
    Similar to `torch.nn.functional.grid_sample` with `align_corners=False`.

    Args:
        im (Tensor): A 2D tensor representing the image.
        i (Tensor): A tensor of i coordinates (in pixel space) at which to interpolate.
        j (Tensor): A tensor of j coordinates (in pixel space) at which to interpolate.

    Returns:
        Tensor: Tensor with the same shape as `i` and `j` containing the interpolated values.
    """

    # Convert coordinates to pixel indices
    h, w = im.shape

    # reshape for indexing purposes
    start_shape = i.shape
    i = i.flatten()
    j = j.flatten()

    # valid
    valid = (i >= -0.5) & (i <= (h - 0.5)) & (j >= -0.5) & (j <= (w - 0.5))

    i0 = backend.long(backend.floor(i))
    j0 = backend.long(backend.floor(j))
    i0 = backend.clamp(i0, 0, h - 2)
    i1 = i0 + 1
    j0 = backend.clamp(j0, 0, w - 2)
    j1 = j0 + 1

    fa = im[i0, j0]
    fb = im[i0, j1]
    fc = im[i1, j0]
    fd = im[i1, j1]

    wa = (i1 - i) * (j1 - j)
    wb = (i1 - i) * (j - j0)
    wc = (i - i0) * (j1 - j)
    wd = (i - i0) * (j - j0)

    result = fa * wa + fb * wb + fc * wc + fd * wd

    if padding_mode == "zeros":
        return (result * valid).reshape(start_shape)
    elif padding_mode == "border":
        return result.reshape(start_shape)
    raise ValueError(f"Unsupported padding mode: {padding_mode}")
