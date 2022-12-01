from .sky_model_object import Sky_Model
import numpy as np
from scipy.stats import iqr
import torch

__all__ = ["Flat_Sky"]

class Flat_Sky(Sky_Model):
    """Model for the sky background in which all values across the image
    are the same.

    """
    model_type = f"flat {Sky_Model.model_type}"
    parameter_specs = {
        "sky": {"units": "flux/arcsec^2"},
    }
    _parameter_order = Sky_Model._parameter_order + ("sky",)

    @torch.no_grad()
    def initialize(self):
        super().initialize()

        if self["sky"].value is None:
            self["sky"].set_representation(
                torch.median(self.target[self.model_image].data) / self.target.pixelscale**2,
                override_locked=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(self.target[self.model_image].data.detach().cpu().numpy(), rng=(31.731 / 2, 100 - 31.731 / 2)) / (2.0 * self.target.pixelscale.item()**2)) / np.sqrt(np.prod(self.fit_window.shape.detach().cpu().numpy())),
                override_locked=True,
            )

    def evaluate_model(self, image):
        
        return torch.ones(image.data.shape, dtype = self.dtype, device = self.device) * (self["sky"].value * image.pixelscale**2)
