from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .star_model_object import Star_Model
from .superellipse_model import SuperEllipse_Galaxy
import torch
import numpy as np
from scipy.stats import iqr
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import matplotlib.pyplot as plt
from scipy.optimize import minimize

__all__ = ["Sersic_Galaxy", "Sersic_Star", "Sersic_Warp", "Sersic_Ray"]

class Sersic_Galaxy(Galaxy_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile.

    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Star(Star_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile.

    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(torch.sqrt(X**2 + Y**2), image)
    def initialize(self):
        super().initialize()
        if self["n"].value is not None and self["flux"].value is not None and self["Rs"].value is not None:
            return
        with torch.no_grad():
            target_area = self.target[self.fit_window]
            self["flux"].set_value(np.log10(np.sum(target_area.data.detach().numpy())), override_locked = self["flux"].value is None)
            self["flux"].set_uncertainty(1e-2, override_locked = self["flux"].uncertainty is None)
            self["n"].set_value(1., override_locked = self["n"].value is None)
            self["n"].set_uncertainty(1e-2, override_locked = self["n"].uncertainty is None)
            self["Rs"].set_value(1., override_locked = self["Rs"].value is None)
            self["Rs"].set_uncertainty(1e-2, override_locked = self["Rs"].uncertainty is None)
    
class Sersic_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a sersic profile for the radial
    light profile.

    """
    model_type = f"sersic {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    
class Sersic_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a sersic profile for the
    radial light model.

    """
    model_type = f"sersic {Warp_Galaxy.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Ray(Ray_Galaxy):
    """ray galaxy model with a sersic profile for the
    radial light model.

    """
    model_type = f"sersic {Ray_Galaxy.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }
    ray_model_parameters = ("I0", "n", "Rs")

    def initialize(self):
        super(self.__class__, self).initialize()
        if all((self["n_0"].value is not None, self["flux_0"].value is not None, self["Rs_0"].value is not None)):
            return
        with torch.no_grad():
            # Get the sub-image area corresponding to the model image
            target_area = self.target[self.fit_window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(
                self["center"].value[0].detach().item(),
                self["center"].value[1].detach().item(), target_area
            )
            iso_info = isophotes(
                target_area.data.detach().numpy() - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().item(), q = self["q"].value.detach().item(),
                n_isophotes = 15,
                more = True,
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
            for r in range(self.rays):
                flux = []
                for iso in iso_info:
                    modangles = (iso["angles"] - (self["PA"].value.detach().item() + r*np.pi/self.rays)) % np.pi
                    flux.append(np.median(iso["isovals"][np.logical_or(modangles < (0.5*np.pi/self.rays), modangles >= (np.pi*(1 - 0.5/self.rays)))]) / self.target.pixelscale**2)
                flux = np.array(flux)
                if np.sum(flux < 0) >= 1:
                    flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
                x0 = [
                    2. if self[f"n_{r}"].value is None else self[f"n_{r}"].value.detach().item(),
                    R[1] if self[f"Rs_{r}"].value is None else self[f"Rs_{r}"].value.detach().item(),
                    flux[0],
                ]
                res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
                self[f"n_{r}"].set_value(res.x[0], override_locked = (self[f"n_{r}"].value is None))
                self[f"Rs_{r}"].set_value(res.x[1], override_locked = (self[f"Rs_{r}"].value is None))
                self[f"flux_{r}"].set_value(np.log10(sersic_I0_to_flux_np(res.x[2], self[f"n_{r}"].value.detach().item(), self[f"Rs_{r}"].value.detach().item(), self["q"].value.detach().item())), override_locked = (self[f"flux_{r}"].value is None))
                if self[f"Rs_{r}"].uncertainty is None:
                    self[f"Rs_{r}"].set_uncertainty(0.02 * self[f"Rs_{r}"].value.detach().item(), override_locked = True)
                if self[f"flux_{r}"].uncertainty is None:
                    self[f"flux_{r}"].set_uncertainty(0.02, override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        return sersic_torch(R, self[f"n_{i}"].value, self[f"Rs_{i}"].value, sersic_flux_to_I0_torch(10**self[f"flux_{i}"].value, self[f"n_{i}"].value, self[f"Rs_{i}"].value, self["q"].value) * sample_image.pixelscale**2)
        
