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

    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["sky"].value is not None:
            self["sky"].set_value(self["sky"].value / self.target.pixelscale**2, override_fixed = True)    
    
    def initialize(self):        
        super().initialize()

        if self["sky"].value is None:
            self["sky"].set_representation(
                np.median(self.target[self.model_image].data) / self.target.pixelscale**2,
                override_fixed=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(self.target[self.model_image].data, rng=(31.731 / 2, 100 - 31.731 / 2)) / (2.0 * self.target.pixelscale**2)) / np.sqrt(np.prod(self.fit_window.shape)),
                override_fixed=True,
            )

    def evaluate_model(self, image):
        
        return torch.ones(image.data.shape) * (self["sky"].value * image.pixelscale**2)
