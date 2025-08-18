import torch
from torch import Tensor
from ...backend_obj import backend, ArrayLike
import numpy as np

from ...param import forward


class RadialMixin:
    """This model defines its `brightness(x,y)` function using a radial model.
    Thus the brightness is instead defined as`radial_model(R)`

    More specifically the function is:

    $$x, y = {\\rm transform\\_coordinates}(x, y)$$
    $$R = {\\rm radius\\_metric}(x, y)$$
    $$I(x, y) = {\\rm radial\\_model}(R)$$

    The `transform_coordinates` function depends on the model. In its simplest
    form it simply subtracts the center of the model to re-center the coordinates.

    The `radius_metric` function is also model dependent, in its simplest form
    this is just $R = \\sqrt{x^2 + y^2}$.
    """

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = self.transform_coordinates(x, y)
        return self.radial_model(self.radius_metric(x, y))


class WedgeMixin:
    """Defines a model with multiple profiles that form wedges projected from the center.

    model which defines multiple radial models separately along some number of
    wedges projected from the center. These wedges have sharp transitions along boundary angles theta.

    **Options:**
    -    `symmetric`: If True, the model will have symmetry for rotations of pi radians
        and each ray will appear twice on the sky on opposite sides of the model.
        If False, each ray is independent.
    -    `segments`: The number of segments to divide the model into. This controls
        how many rays are used in the model. The default is 2
    """

    _model_type = "wedge"
    _options = ("segments", "symmetric")

    def __init__(self, *args, symmetric: bool = True, segments: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R: ArrayLike, T: ArrayLike) -> ArrayLike:
        model = backend.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        angles = (T + w / 2) % cycle
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            indices = (angles >= v[s]) & (angles < (v[s] + w))
            model = backend.add_at_indices(model, indices, self.iradial_model(s, R[indices]))
        return model

    def brightness(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))


class RayMixin:
    """Defines a model with multiple profiles along rays projected from the center.

    model which defines multiple radial models separately along some number of
    rays projected from the center. These rays smoothly transition from one to
    another along angles theta. The ray transition uses a cosine smoothing
    function which depends on the number of rays, for example with two rays the
    brightness would be:

    $$I(R,\\theta) = I_1(R)*\\cos(\\theta \\% \\pi) + I_2(R)*\\cos((\\theta + \\pi/2) \\% \\pi)$$

    For $\\theta = 0$ the brightness comes entirely from `I_1` while for $\\theta = \\pi/2$
    the brightness comes entirely from `I_2`.

    **Options:**
    -    `symmetric`: If True, the model will have symmetry for rotations of pi radians
        and each ray will appear twice on the sky on opposite sides of the model.
        If False, each ray is independent.
    -    `segments`: The number of segments to divide the model into. This controls
        how many rays are used in the model. The default is 2
    """

    _model_type = "ray"
    _options = ("symmetric", "segments")

    def __init__(self, *args, symmetric: bool = True, segments: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        self.segments = segments

    def polar_model(self, R: ArrayLike, T: ArrayLike) -> ArrayLike:
        model = backend.zeros_like(R)
        weight = backend.zeros_like(R)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            angles = (T + cycle / 2 - v[s]) % cycle - cycle / 2
            indices = (angles >= -w) & (angles < w)
            weights = (backend.cos(angles[indices] * self.segments) + 1) / 2
            model = backend.add_at_indices(
                model, indices, weights * self.iradial_model(s, R[indices])
            )
            weight = backend.add_at_indices(weight, indices, weights)
        return model / weight

    def brightness(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        x, y = self.transform_coordinates(x, y)
        return self.polar_model(self.radius_metric(x, y), self.angular_metric(x, y))
