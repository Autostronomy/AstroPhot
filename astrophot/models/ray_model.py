import numpy as np
import torch

from .galaxy_model_object import GalaxyModel

__all__ = ["RayGalaxy"]


class RayGalaxy(GalaxyModel):
    """Variant of a galaxy model which defines multiple radial models
    seprarately along some number of rays projected from the galaxy
    center. These rays smoothly transition from one to another along
    angles theta. The ray transition uses a cosine smoothing function
    which depends on the number of rays, for example with two rays the
    brightness would be:

    I(R,theta) = I1(R)*cos(theta % pi) + I2(R)*cos((theta + pi/2) % pi)

    Where I(R,theta) is the brightness function in polar coordinates,
    R is the semi-major axis, theta is the polar angle (defined after
    galaxy axis ratio is applied), I1(R) is the first brightness
    profile, % is the modulo operator, and I2 is the second brightness
    profile. The ray model defines no extra parameters, though now
    every model parameter related to the brightness profile gains an
    extra dimension for the ray number. Also a new input can be given
    when instantiating the ray model: "rays" which is an integer for
    the number of rays.

    """

    _model_type = "segments"
    usable = False
    _options = ("symmetric_rays", "rays")

    def __init__(self, *args, symmetric_rays=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric_rays = symmetric_rays
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        if self.segments % 2 == 0 and self.symmetric_rays:
            for r in range(self.segments):
                angles = (T - (r * np.pi / self.segments)) % np.pi
                indices = torch.logical_or(
                    angles < (np.pi / self.segments),
                    angles >= (np.pi * (1 - 1 / self.segments)),
                )
                weight = (torch.cos(angles[indices] * self.segments) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices])
        elif self.segments % 2 == 1 and self.symmetric_rays:
            for r in range(self.segments):
                angles = (T - (r * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / self.segments),
                    angles >= (np.pi * (2 - 1 / self.segments)),
                )
                weight = (torch.cos(angles[indices] * self.segments) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices])
                angles = (T - (np.pi + r * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / self.segments),
                    angles >= (np.pi * (2 - 1 / self.segments)),
                )
                weight = (torch.cos(angles[indices] * self.segments) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices])
        elif self.segments % 2 == 0 and not self.symmetric_rays:
            for r in range(self.segments):
                angles = (T - (r * 2 * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (2 * np.pi / self.segments),
                    angles >= (2 * np.pi * (1 - 1 / self.segments)),
                )
                weight = (torch.cos(angles[indices] * self.segments) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices])
        else:
            for r in range(self.segments):
                angles = (T - (r * 2 * np.pi / self.segments)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (2 * np.pi / self.segments),
                    angles >= (np.pi * (2 - 1 / self.segments)),
                )
                weight = (torch.cos(angles[indices] * self.segments) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices])
        return model

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))
