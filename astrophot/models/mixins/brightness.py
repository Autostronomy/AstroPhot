import numpy as np


class RadialMixin:

    def brightness(self, x, y, center):
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = x - center[0], y - center[1]
        return self.radial_model(self.radius_metric(x, y))


def rotate(theta, x, y):
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y


class InclinedMixin:

    parameter_specs = {
        "q": {"units": "b/a", "limits": (0, 1), "uncertainty": 0.03},
        "PA": {
            "units": "radians",
            "limits": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.06,
        },
    }

    def brightness(self, x, y, center, PA, q):
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = x - center[0], y - center[1]
        x, y = rotate(PA, x, y)
        return self.radial_model((x**2 + (y / q) ** 2).sqrt())
