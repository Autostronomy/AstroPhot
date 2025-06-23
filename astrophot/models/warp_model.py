import numpy as np
import torch

from .galaxy_model_object import GalaxyModel
from ..utils.interpolate import default_prof
from ..utils.decorators import ignore_numpy_warnings
from . import func
from ..param import forward

__all__ = ["WarpGalaxy"]


class WarpGalaxy(GalaxyModel):
    """Galaxy model which includes radially varrying PA and q
    profiles. This works by warping the coordinates using the same
    transform for a global PA/q except applied to each pixel
    individually. In the limit that PA and q are a constant, this
    recovers a basic galaxy model with global PA/q. However, a linear
    PA profile will give a spiral appearance, variations of PA/q
    profiles can create complex galaxy models. The form of the
    coordinate transformation looks like:

    X, Y = meshgrid(image)
    R = sqrt(X^2 + Y^2)
    X', Y' = Rot(theta(R), X, Y)
    Y'' = Y' / q(R)

    where the definitions are the same as for a regular galaxy model,
    except now the theta is a function of radius R (before
    transformation) and the axis ratio q is also a function of radius
    (before the transformation).

    Parameters:
        q(R): Tensor of axis ratio values for axis ratio spline
        PA(R): Tensor of position angle values as input to the spline

    """

    _model_type = "warp"
    _parameter_specs = {
        "q_R": {"units": "b/a", "valid": (0.0, 1), "uncertainty": 0.04},
        "PA_R": {
            "units": "radians",
            "valid": (0, np.pi),
            "cyclic": True,
            "uncertainty": 0.08,
        },
    }
    usable = False

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.PA_R.value is None:
            if self.PA_R.prof is None:
                self.PA_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.PA_R.dynamic_value = np.zeros(len(self.PA_R.prof)) + np.pi / 2
            self.PA_R.uncertainty = (10 * np.pi / 180) * torch.ones_like(self.PA_R.value)
        if self.q_R.value is None:
            if self.q_R.prof is None:
                self.q_R.prof = default_prof(self.window.shape, self.target.pixel_length, 2, 0.2)
            self.q_R.dynamic_value = np.ones(len(self.q_R.prof)) * 0.8
            self.q_R.uncertainty = self.default_uncertainty * self.q_R.value

    @forward
    def transform_coordinates(self, x, y, q_R, PA_R):
        x, y = super().transform_coordinates(x, y)
        R = self.radius_metric(x, y)
        PA = func.spline(R, self.PA_R.prof, PA_R)
        q = func.spline(R, self.q_R.prof, q_R)
        x, y = func.rotate(PA, x, y)
        return x, y / q
