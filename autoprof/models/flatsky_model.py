from .sky_model_object import Sky_Model
import numpy as np
from scipy.stats import iqr
import torch

class FlatSky(Sky_Model):

    model_type = f"flat {Sky_Model.model_type}"
    parameter_specs = {
        "sky": {"units": "flux/arcsec^2"},
        "noise": {"units": "flux/arcsec^2", "limits": (0,None)},
    }

    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["sky"].value is not None:
            self["sky"].set_value(self["sky"].value / self.target.pixelscale**2, override_fixed = True)    
        if self["noise"].value is not None:
            self["noise"].set_value(self["noise"].value / self.target.pixelscale**2, override_fixed = True)    
    
    def initialize(self):        
        super().initialize()

        if self["sky"].value is None:
            self["sky"].set_representation(
                np.median(self.target[self.model_image].data) / self.target.pixelscale**2,
                override_fixed=True,
            )
        if self["noise"].value is None:
            self["noise"].set_representation(
                iqr(self.target[self.model_image].data, rng=(31.731 / 2, 100 - 31.731 / 2)) / (2.0 * self.target.pixelscale**2),
                override_fixed=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                self["noise"].value / np.sqrt(np.prod(self.fit_window.shape)),
                override_fixed=True,
            )
        if self["noise"].uncertainty is None:
            self["noise"].set_uncertainty(
                self["noise"].value / np.sqrt(2 * np.prod(self.fit_window.shape) - 2),
                override_fixed=True,
            )

    def evaluate_model(self, image):
        
        return torch.ones(image.data.shape) * (self["sky"].value * image.pixelscale**2)
