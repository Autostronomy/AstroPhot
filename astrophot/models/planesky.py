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

    .. math::

       I(X, Y) = I_0 + X*\\delta_x + Y*\\delta_y

    where :math:`I(X,Y)` is the brightness as a function of image position :math:`X, Y`,
    :math:`I_0` is the central sky brightness value, and :math:`\\delta_x, \\delta_y` are the slopes of
    the sky brightness plane.

    :param I0: central sky brightness value
    :param delta: Tensor for slope of the sky brightness in each image dimension

    """

    _model_type = "plane"
    _parameter_specs = {
        "I0": {
            "units": "flux/arcsec^2",
            "shape": (),
            "dynamic": True,
            "description": "central sky brightness value",
        },
        "delta": {
            "units": "flux/arcsec",
            "shape": (2,),
            "dynamic": True,
            "description": "Tensor for slope of the sky brightness in each image dimension",
        },
    }
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not self.I0.initialized:
            dat = backend.to_numpy(self.target[self.window]._data).copy()
            mask = backend.to_numpy(self.target[self.window]._mask)
            dat[mask] = np.median(dat[~mask])
            self.I0.value = np.median(dat) / self.target.pixel_area.item()
        if not self.delta.initialized:
            self.delta.value = [0.0, 0.0]

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I0: ArrayLike, delta: ArrayLike) -> ArrayLike:
        return I0 + x * delta[0] + y * delta[1]
