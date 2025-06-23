import torch

from ...param import forward
from ...utils.decorators import ignore_numpy_warnings
from .._shared_methods import _sample_image
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
            prof = [0, 2 * target_area.pixel_length]
            while prof[-1] < (max(self.window.shape) * target_area.pixel_length / 2):
                prof.append(prof[-1] + torch.max(2 * target_area.pixel_length, prof[-1] * 0.2))
            prof.pop()
            prof.append(
                torch.sqrt(
                    torch.sum((self.window.shape[0] / 2) ** 2 + (self.window.shape[1] / 2) ** 2)
                    * target_area.pixel_length**2
                )
            )
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
            prof = [0, 2 * target_area.pixel_length]
            while prof[-1] < (max(self.window.shape) * target_area.pixel_length / 2):
                prof.append(prof[-1] + torch.max(2 * target_area.pixel_length, prof[-1] * 0.2))
            prof.pop()
            prof.append(
                torch.sqrt(
                    torch.sum((self.window.shape[0] / 2) ** 2 + (self.window.shape[1] / 2) ** 2)
                    * target_area.pixel_length**2
                )
            )
            self.I_R.prof = [prof] * self.segments
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
    def iradial_model(self, i, R, I_R):
        return func.spline(R, self.I_R.prof[i], I_R[i])
