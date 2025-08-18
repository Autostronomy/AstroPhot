import numpy as np
import torch

from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from .sky_model_object import SkyModel
from ..backend_obj import backend, ArrayLike
from ..param import forward

__all__ = ["FlatSky"]


@combine_docstrings
class FlatSky(SkyModel):
    """Model for the sky background in which all values across the image
    are the same.

    **Parameters:**
    -    `I`: brightness for the sky, represented as the log of the brightness over pixel scale squared, this is proportional to a surface brightness

    """

    _model_type = "flat"
    _parameter_specs = {
        "I": {"units": "flux/arcsec^2"},
    }
    usable = True

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I.initialized:
            return

        dat = backend.to_numpy(self.target[self.window].data).copy()
        self.I.dynamic_value = np.median(dat) / self.target.pixel_area.item()

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I: ArrayLike) -> ArrayLike:
        return backend.ones_like(x) * I
