from .sky_model_object import Sky_Model
import numpy as np
from scipy.stats import iqr
import torch

__all__ = ["Plane_Sky"]

class Plane_Sky(Sky_Model):
    """Sky background model using a tilted plane for the sky flux. The brightness for each pixel is defined as:

    I(X, Y) = S + X*dx + Y*dy

    where I(X,Y) is the brightness as a funcion of image position X Y,
    S is the central sky brightness value, and dx dy are the slopes of
    the sky brightness plane.

    Parameters:
        sky: central sky brightness value
        delta: Tensor for slope of the sky brightness in each image dimension

    """
    model_type = f"plane {Sky_Model.model_type}"
    parameter_specs = {
        "sky": {"units": "flux/arcsec^2"},
        "delta": {"units": "sky/arcsec"},
    }
    _parameter_order = Sky_Model._parameter_order + ("sky", "delta")

    @torch.no_grad()
    def initialize(self, target = None):        
        if target is None:
            target = self.target
        super().initialize(target)

        if self["sky"].value is None:
            self["sky"].set_value(
                np.median(target[self.window].data) / target.pixelscale**2,
                override_locked=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(target[self.window].data, rng=(31.731 / 2, 100 - 31.731 / 2)) / (2.0 * target.pixelscale**2)) / np.sqrt(np.prod(self.window.shape.detach().cpu().numpy())),
                override_locked=True,
            )
        if self["delta"].value is None:
            self["delta"].set_value([0., 0.], override_locked = True)
            self["delta"].set_uncertainty([0.1,0.1], override_locked = True)
            
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return (self["sky"].value * image.pixelscale**2) + X*self["delta"].value[0] + Y*self["delta"].value[1]
