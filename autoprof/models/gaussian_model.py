from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .star_model_object import Star_Model
import torch
import numpy as np

class Gaussian_Galaxy(Galaxy_Model):
    """Basic galaxy model with Gaussian as the radial light profile.

    """
    model_type = f"gaussian {Galaxy_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize
    
class Gaussian_Warp(Warp_Galaxy):
    """Coordinate warped galaxy model with Gaussian as the radial light
    profile.

    """
    model_type = f"gaussian {Warp_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize


class Gaussian_Star(Star_Model):
    """Basica star model with a Gaussian as the radial light profile.

    """
    model_type = f"gaussian {Star_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }

    def initialize(self):
        super().initialize()
        if self["sigma"].value is not None and self["flux"].value is not None:
            return
        with torch.no_grad():
            target_area = self.target[self.fit_window]
            self["sigma"].set_value(1, override_locked = self["sigma"].value is None)
            self["sigma"].set_uncertainty(1e-2, override_locked = self["sigma"].uncertainty is None)
            self["flux"].set_value(np.sum(target_area.data.detach().numpy()), override_locked = self["flux"].value is None)
            self["flux"].set_uncertainty(self["flux"].value.detach().numpy() * 1e-2, override_locked = self["flux"].uncertainty is None)
    from ._shared_methods import gaussian_radial_model as radial_model

    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(torch.sqrt(X**2 + Y**2), image)
        
    
