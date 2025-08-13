import torch
from torch import Tensor
import numpy as np

from ...param import forward
from ...backend_obj import ArrayLike
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import _sample_image
from ...utils.interpolate import default_prof
from .. import func


class SplineMixin:
    """Spline radial model for brightness.

    The `radial_model` function for this model is defined as a spline
    interpolation from the parameter `I_R`. The `I_R` parameter is a tensor
    that contains the radial profile of the brightness in units of
    flux/arcsec^2. The radius of each node is determined from `I_R.prof`.

    **Parameters:**
    -    `I_R`: Tensor of radial brightness values in units of flux/arcsec^2.
    """

    _model_type = "spline"
    _parameter_specs = {"I_R": {"units": "flux/arcsec^2", "valid": (0, None)}}

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I_R.initialized:
            return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixelscale, 2, 0.2)
            prof = np.append(prof, prof[-1] * 10)
            self.I_R.prof = prof
        else:
            prof = self.I_R.prof

        R, I, S = _sample_image(
            target_area,
            self.transform_coordinates,
            self.radius_metric,
            rad_bins=[0] + list((prof[:-1] + prof[1:]) / 2) + [prof[-1] * 100],
        )
        self.I_R.dynamic_value = 10**I

    @forward
    def radial_model(self, R: ArrayLike, I_R: ArrayLike) -> ArrayLike:
        ret = func.spline(R, self.I_R.prof, I_R)
        return ret


class iSplineMixin:
    """Batched spline radial model for brightness.

    The `radial_model` function for this model is defined as a spline
    interpolation from the parameter `I_R`. The `I_R` parameter is a tensor that
    contains the radial profile of the brightness in units of flux/arcsec^2. The
    radius of each node is determined from `I_R.prof`.

    Both `I_R` and `I_R.prof` are batched by their first dimension, allowing for
    multiple spline profiles to be defined at once. Each individual spline model
    is then `I_R[i]` and `I_R.prof[i]` where `i` indexes the profiles.

    **Parameters:**
    -    `I_R`: Tensor of radial brightness values in units of flux/arcsec^2.
    """

    _model_type = "spline"
    _parameter_specs = {"I_R": {"units": "flux/arcsec^2", "valid": (0, None)}}

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I_R.initialized:
            return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixelscale, 2, 0.2)
            prof = np.append(prof, prof[-1] * 10)
            prof = np.stack([prof] * self.segments)
            self.I_R.prof = prof
        else:
            prof = self.I_R.prof

        value = np.zeros(prof.shape)
        cycle = np.pi if self.symmetric else 2 * np.pi
        w = cycle / self.segments
        v = w * np.arange(self.segments)
        for s in range(self.segments):
            angle_range = (v[s] - w / 2, v[s] + w / 2)
            R, I, S = _sample_image(
                target_area,
                self.transform_coordinates,
                self.radius_metric,
                angle=self.angular_metric,
                rad_bins=[0] + list((prof[s][:-1] + prof[s][1:]) / 2) + [prof[s][-1] * 100],
                angle_range=angle_range,
                cycle=cycle,
            )
            value[s] = I

        self.I_R.dynamic_value = 10**value

    @forward
    def iradial_model(self, i: int, R: ArrayLike, I_R: ArrayLike) -> ArrayLike:
        return func.spline(R, self.I_R.prof[i], I_R[i])
