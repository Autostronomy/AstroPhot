from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .star_model_object import Star_Model
import torch
import numpy as np

__all__ = ["Gaussian_Galaxy", "Gaussian_Warp", "Gaussian_Star"]

class Gaussian_Galaxy(Galaxy_Model):
    """Basic galaxy model with Gaussian as the radial light profile.

    """
    model_type = f"gaussian {Galaxy_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }
    parameter_order = Galaxy_Model.parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_SuperEllipse(SuperEllipse_Galaxy):
    """Super ellipse galaxy model with Gaussian as the radial light
    profile.

    """
    model_type = f"gaussian {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }
    parameter_order = SuperEllipse_Galaxy.parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a gaussian profile for the
    radial light profile.

    """
    model_type = f"gaussian {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }
    parameter_order = SuperEllipse_Warp.parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize
    
class Gaussian_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile.

    """
    model_type = f"gaussian {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }
    parameter_order = FourierEllipse_Galaxy.parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile.

    """
    model_type = f"gaussian {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "flux", "limits": (0,None)},
    }
    parameter_order = FourierEllipse_Warp.parameter_order + ("sigma", "flux")

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
    parameter_order = Warp_Galaxy.parameter_order + ("sigma", "flux")

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
    parameter_order = Star_Model.parameter_order + ("sigma", "flux")

    def initialize(self):
        super().initialize()
        if self["sigma"].value is not None and self["flux"].value is not None:
            return
        with torch.no_grad():
            target_area = self.target[self.fit_window]
            self["sigma"].set_value(1, override_locked = self["sigma"].value is None)
            self["sigma"].set_uncertainty(1e-2, override_locked = self["sigma"].uncertainty is None)
            self["flux"].set_value(np.sum(target_area.data.detach().cpu().numpy()), override_locked = self["flux"].value is None)
            self["flux"].set_uncertainty(self["flux"].value.detach().cpu().numpy() * 1e-2, override_locked = self["flux"].uncertainty is None)
    from ._shared_methods import gaussian_radial_model as radial_model

    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(torch.sqrt(X**2 + Y**2), image)
        
    
