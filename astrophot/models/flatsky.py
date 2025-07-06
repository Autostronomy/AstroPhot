import numpy as np
from scipy.stats import iqr
import torch

from ..utils.decorators import ignore_numpy_warnings
from .sky_model_object import SkyModel
from ..param import forward

__all__ = ["FlatSky"]


class FlatSky(SkyModel):
    """Model for the sky background in which all values across the image
    are the same.

    Parameters:
        I: brightness for the sky, represented as the log of the brightness over pixel scale squared, this is proportional to a surface brightness

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
        self.I.value = np.median(dat) / self.target.pixel_area.item()
        self.I.uncertainty = (
            iqr(dat, rng=(16, 84)) / (2.0 * self.target.pixel_area.item())
        ) / np.sqrt(np.prod(self.window.shape))

    @forward
    def brightness(self, x, y, I):
        return torch.ones_like(x) * I
