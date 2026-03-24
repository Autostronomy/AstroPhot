from typing import Tuple
from ...backend_obj import backend, ArrayLike


def rotate(theta: ArrayLike, x: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = backend.sin(theta)
    c = backend.cos(theta)
    return c * x - s * y, s * x + c * y
