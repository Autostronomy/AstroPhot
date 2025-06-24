import numpy as np
from scipy.stats import iqr
import torch

from .sky_model_object import SkyModel
from ..utils.decorators import ignore_numpy_warnings
from ..param import forward

__all__ = ["PlaneSky"]


class PlaneSky(SkyModel):
    """Sky background model using a tilted plane for the sky flux. The brightness for each pixel is defined as:

    I(X, Y) = S + X*dx + Y*dy

    where I(X,Y) is the brightness as a function of image position X Y,
    S is the central sky brightness value, and dx dy are the slopes of
    the sky brightness plane.

    Parameters:
        sky: central sky brightness value
        delta: Tensor for slope of the sky brightness in each image dimension

    """

    _model_type = "plane"
    _parameter_specs = {
        "I0": {"units": "flux/arcsec^2"},
        "delta": {"units": "flux/arcsec"},
    }
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I0.value is None:
            self.I0.dynamic_value = (
                np.median(self.target[self.window].data.npvalue) / self.target.pixel_area.item()
            )
            self.I0.uncertainty = (
                iqr(
                    self.target[self.window].data.npvalue,
                    rng=(16, 84),
                )
                / 2.0
            ) / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy()))
        if self.delta.value is None:
            self.delta.dynamic_value = [0.0, 0.0]
            self.delta.uncertainty = [
                self.default_uncertainty,
                self.default_uncertainty,
            ]

    @forward
    def brightness(self, x, y, I0, delta):
        return I0 + x * delta[0] + y * delta[1]
