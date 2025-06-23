import torch
import numpy as np

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import _sample_image
from ...utils.interpolate import default_prof
from .. import func


class SplineMixin:

    _model_type = "spline"
    parameter_specs = {
        "I_R": {"units": "flux/arcsec^2"},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I_R.value is not None:
            return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixel_length, 2, 0.2)
            self.I_R.prof = prof
        else:
            prof = self.I_R.prof

        R, I, S = _sample_image(
            target_area,
            self.transform_coordinates,
            self.radius_metric,
            rad_bins=[0] + list((prof[:-1] + prof[1:]) / 2) + [prof[-1] * 100],
        )
        self.I_R.dynamic_value = I
        self.I_R.uncertainty = S

    @forward
    def radial_model(self, R, I_R):
        return func.spline(R, self.I_R.prof, I_R)


class iSplineMixin:

    _model_type = "spline"
    parameter_specs = {
        "I_R": {"units": "flux/arcsec^2"},
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I_R.value is not None:
            return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixel_length, 2, 0.2)
            self.I_R.prof = [prof] * self.segments
        else:
            prof = self.I_R.prof

        value = np.zeros((self.segments, len(prof)))
        uncertainty = np.zeros((self.segments, len(prof)))
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
            )
            value[s] = I
            uncertainty[s] = S
        self.I_R.dynamic_value = value
        self.I_R.uncertainty = uncertainty

    @forward
    def iradial_model(self, i, R, I_R):
        return func.spline(R, self.I_R.prof[i], I_R[i])
