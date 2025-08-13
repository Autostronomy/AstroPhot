import numpy as np
import torch

from .sky_model_object import SkyModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..param import forward
from ..backend_obj import backend, ArrayLike

__all__ = ["PlaneSky"]


@combine_docstrings
class PlaneSky(SkyModel):
    """Sky background model using a tilted plane for the sky flux. The brightness for each pixel is defined as:

    $$I(X, Y) = I_0 + X*\\delta_x + Y*\\delta_y$$

    where $I(X,Y)$ is the brightness as a function of image position $X, Y$,
    $I_0$ is the central sky brightness value, and $\\delta_x, \\delta_y$ are the slopes of
    the sky brightness plane.

    **Parameters:**
    -    `I0`: central sky brightness value
    -    `delta`: Tensor for slope of the sky brightness in each image dimension

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

        if not self.I0.initialized:
            dat = backend.to_numpy(self.target[self.window].data).copy()
            self.I0.dynamic_value = np.median(dat) / self.target.pixel_area.item()
        if not self.delta.initialized:
            self.delta.dynamic_value = [0.0, 0.0]

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I0: ArrayLike, delta: ArrayLike) -> ArrayLike:
        return I0 + x * delta[0] + y * delta[1]
