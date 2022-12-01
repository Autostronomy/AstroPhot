from .sky_model_object import Sky_Model
import numpy as np
from scipy.stats import iqr
import torch

__all__ = ["Plane_Sky"]

class Plane_Sky(Sky_Model):
    """Sky background model using a tilted plane for the sky flux.

    """
    model_type = f"plane {Sky_Model.model_type}"
    parameter_specs = {
        "sky": {"units": "flux/arcsec^2"},
        "delta": {"units": "sky/arcsec"},
    }
    _parameter_order = Sky_Model._parameter_order + ("sky", "delta")

    def initialize(self):        
        super().initialize()

        if self["sky"].value is None:
            self["sky"].set_value(
                np.median(self.target[self.model_image].data) / self.target.pixelscale**2,
                override_locked=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(self.target[self.model_image].data, rng=(31.731 / 2, 100 - 31.731 / 2)) / (2.0 * self.target.pixelscale**2)) / np.sqrt(np.prod(self.fit_window.shape)),
                override_locked=True,
            )
        if self["delta"].value is None:
            self["delta"].set_value([0., 0.], override_locked = True)
            self["delta"].set_uncertainty([0.1,0.1], override_locked = True)
            
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return (self["sky"].value * image.pixelscale**2) + X*self["delta"].value[0] + Y*self["delta"].value[1]
