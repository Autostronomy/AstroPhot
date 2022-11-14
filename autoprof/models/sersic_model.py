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
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Star(Star_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile.

    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(self.radius_metric(X, Y), image)
    
class Sersic_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a sersic profile for the radial
    light profile.

    """
    model_type = f"sersic {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    
class Sersic_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a sersic profile for the
    radial light model.

    """
    model_type = f"sersic {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Ray(Ray_Galaxy):
    """ray galaxy model with a sersic profile for the
    radial light model.

    """
    model_type = f"sersic {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    ray_model_parameters = ("Ie", "n", "Re")

    def initialize(self):
        super(self.__class__, self).initialize()
        if all((self["n_0"].value is not None, self["Ie_0"].value is not None, self["Re_0"].value is not None)):
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
                    R[4] if self[f"Re_{r}"].value is None else self[f"Re_{r}"].value.detach().item(),
                    flux[4],
                ]
                res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
                self[f"n_{r}"].set_value(res.x[0], override_locked = (self[f"n_{r}"].value is None))
                self[f"Re_{r}"].set_value(res.x[1], override_locked = (self[f"Re_{r}"].value is None))
                self[f"Ie_{r}"].set_value(np.log10(res.x[2]), override_locked = (self[f"Ie_{r}"].value is None))
                if self[f"Re_{r}"].uncertainty is None:
                    self[f"Re_{r}"].set_uncertainty(0.02 * self[f"Re_{r}"].value.detach().item(), override_locked = True)
                if self[f"Ie_{r}"].uncertainty is None:
                    self[f"Ie_{r}"].set_uncertainty(0.02, override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        return sersic_torch(R, self[f"n_{i}"].value, self[f"Re_{i}"].value, (10**self[f"Ie_{i}"].value) * sample_image.pixelscale**2)
        
