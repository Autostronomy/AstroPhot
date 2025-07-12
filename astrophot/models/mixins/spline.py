import torch
import numpy as np

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import _sample_image
from ...utils.interpolate import default_prof
from .. import func


class SplineMixin:

    _model_type = "spline"
    _parameter_specs = {"I_R": {"units": "flux/arcsec^2"}}
    _overload_parameter_specs = {
        "logI_R": {
            "units": "log10(flux/arcsec^2)",
            "overloads": "I_R",
            "overload_function": lambda p: 10**p.logI_R.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        try:
            if self.logI_R.initialized:
                return
        except AttributeError:
            if self.I_R.initialized:
                return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixelscale, 2, 0.2)
            self.I_R.prof = prof
        else:
            prof = self.I_R.prof

        R, I, S = _sample_image(
            target_area,
            self.transform_coordinates,
            self.radius_metric,
            rad_bins=[0] + list((prof[:-1] + prof[1:]) / 2) + [prof[-1] * 100],
        )
        try:
            self.logI_R.dynamic_value = I
        except AttributeError:
            self.I_R.dynamic_value = 10**I

    @forward
    def radial_model(self, R, I_R):
        return func.spline(R, self.I_R.prof, I_R)


class iSplineMixin:

    _model_type = "spline"
    _parameter_specs = {"I_R": {"units": "flux/arcsec^2"}}
    _overload_parameter_specs = {
        "logI_R": {
            "units": "log10(flux/arcsec^2)",
            "overloads": "I_R",
            "overload_function": lambda p: 10**p.logI_R.value,
        }
    }

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        try:
            if self.logI_R.initialized:
                return
        except AttributeError:
            if self.I_R.initialized:
                return

        target_area = self.target[self.window]
        # Create the I_R profile radii if needed
        if self.I_R.prof is None:
            prof = default_prof(self.window.shape, target_area.pixelscale, 2, 0.2)
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

        if hasattr(self, "logI_R"):
            self.logI_R.dynamic_value = value
        else:
            self.I_R.dynamic_value = 10**value

    @forward
    def iradial_model(self, i, R, I_R):
        return func.spline(R, self.I_R.prof[i], I_R[i])
