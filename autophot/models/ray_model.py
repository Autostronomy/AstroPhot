import numpy as np
import torch

from .galaxy_model_object import Galaxy_Model
from .parameter_object import Parameter
from ..utils.interpolate import cubic_spline_torch
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.conversions.coordinates import Axis_Ratio_Cartesian

__all__ = ["Ray_Galaxy"]


class Ray_Galaxy(Galaxy_Model):
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

    model_type = f"ray {Galaxy_Model.model_type}"
    special_kwargs = Galaxy_Model.special_kwargs + ["rays"]
    rays = 2
    track_attrs = Galaxy_Model.track_attrs + ["rays"]
    useable = False

    def __init__(self, *args, **kwargs):
        self.symmetric_rays = True
        super().__init__(*args, **kwargs)
        self.rays = kwargs.get("rays", Ray_Galaxy.rays)

    @default_internal
    def polar_model(self, R, T, image=None, parameters=None):
        model = torch.zeros_like(R)
        if self.rays % 2 == 0 and self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r * np.pi / self.rays)) % np.pi
                indices = torch.logical_or(
                    angles < (np.pi / self.rays),
                    angles >= (np.pi * (1 - 1 / self.rays)),
                )
                weight = (torch.cos(angles[indices] * self.rays) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        elif self.rays % 2 == 1 and self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r * np.pi / self.rays)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / self.rays),
                    angles >= (np.pi * (2 - 1 / self.rays)),
                )
                weight = (torch.cos(angles[indices] * self.rays) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
                angles = (T - (np.pi + r * np.pi / self.rays)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (np.pi / self.rays),
                    angles >= (np.pi * (2 - 1 / self.rays)),
                )
                weight = (torch.cos(angles[indices] * self.rays) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        elif self.rays % 2 == 0 and not self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r * 2 * np.pi / self.rays)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (2 * np.pi / self.rays),
                    angles >= (2 * np.pi * (1 - 1 / self.rays)),
                )
                weight = (torch.cos(angles[indices] * self.rays) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        else:
            for r in range(self.rays):
                angles = (T - (r * 2 * np.pi / self.rays)) % (2 * np.pi)
                indices = torch.logical_or(
                    angles < (2 * np.pi / self.rays),
                    angles >= (np.pi * (2 - 1 / self.rays)),
                )
                weight = (torch.cos(angles[indices] * self.rays) + 1) / 2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        return model

    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
        XX, YY = self.transform_coordinates(X, Y, image, parameters)

        return self.polar_model(
            self.radius_metric(XX, YY, image=image, parameters=parameters),
            self.angular_metric(XX, YY, image=image, parameters=parameters),
            image=image,
            parameters=parameters,
        )


# class SingleRay_Galaxy(Galaxy_Model):
