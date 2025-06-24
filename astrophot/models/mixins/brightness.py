import torch
import numpy as np

from ...param import forward


class RadialMixin:

    @forward
    def brightness(self, x, y):
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = self.transform_coordinates(x, y)
        return self.radial_model(self.radius_metric(x, y))


class WedgeMixin:
    """Variant of the ray model where no smooth transition is performed
    between regions as a function of theta, instead there is a sharp
    trnasition boundary. This may be desirable as it cleanly
    separates where the pixel information is going. Due to the sharp
    transition though, it may cause unusual behaviour when fitting. If
    problems occur, try fitting a ray model first then fix the center,
    PA, and q and then fit the wedge model. Essentially this breaks
    down the structure fitting and the light profile fitting into two
    steps. The wedge model, like the ray model, defines no extra
    parameters, however a new option can be supplied on instantiation
    of the wedge model which is "wedges" or the number of wedges in
    the model.

    """

    _model_type = "wedge"
    _options = ("segments", "symmetric")

    def __init__(self, *args, symmetric=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        angles = (T + w / 2) % cycle
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            indices = (angles >= v[s]) & (angles < (v[s] + w))
            model[indices] += self.iradial_model(s, R[indices])
        return model

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))


class RayMixin:
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

    _model_type = "ray"
    _options = ("symmetric", "segments")

    def __init__(self, *args, symmetric=True, segments=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R, T):
        model = torch.zeros_like(R)
        weight = torch.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            angles = (T + cycle / 2 - v[s]) % cycle - cycle / 2
            indices = (angles >= -w) & (angles < w)
            weights = (torch.cos(angles[indices] * self.segments) + 1) / 2
            model[indices] += weights * self.iradial_model(s, R[indices])
            weight[indices] += weights
        return model / weight

    def brightness(self, x, y):
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))
