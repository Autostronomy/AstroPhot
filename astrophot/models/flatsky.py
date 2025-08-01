import numpy as np
import torch
from torch import Tensor

from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from .sky_model_object import SkyModel
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

        dat = self.target[self.window].data.detach().cpu().numpy().copy()
        self.I.dynamic_value = np.median(dat) / self.target.pixel_area.item()

    @forward
    def brightness(self, x: Tensor, y: Tensor, I: Tensor) -> Tensor:
        return torch.ones_like(x) * I
