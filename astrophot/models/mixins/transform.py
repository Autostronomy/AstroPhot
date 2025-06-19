import numpy as np
from ...param import forward


def rotate(theta, x, y):
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y


class InclinedMixin:

    _parameter_specs = {
        "q": {"units": "b/a", "valid": (0, 1), "uncertainty": 0.03, "shape": ()},
        "PA": {
            "units": "radians",
            "valid": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
            "shape": (),
        },
    }

    @forward
    def transform_coordinates(self, x, y, PA, q):
        """
        Transform coordinates based on the position angle and axis ratio.
        """
        x, y = super().transform_coordinates(x, y)
        x, y = rotate(-(PA + np.pi / 2), x, y)
        return x, y / q
