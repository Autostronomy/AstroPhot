from typing import Tuple
from torch import Tensor


def rotate(theta: Tensor, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y
