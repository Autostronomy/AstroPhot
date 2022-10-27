from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
import torch
import numpy as np
from scipy.stats import iqr
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Sersic_Galaxy(Galaxy_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile.

    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
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
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
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
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }
    ray_model_parameters = ("I0", "n", "Rs")

    def initialize(self):
        super(self.__class__, self).initialize()
        if all((self["n_0"].value is not None, self["I0_0"].value is not None, self["Rs_0"].value is not None)):
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
                    flux[0] if self[f"I0_{r}"].value is None else self[f"I0_{r}"].value.detach().item(),
                ]
                res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
                plt.scatter(R, np.log10(flux))
                plt.plot(R, np.log10(sersic_np(R, res.x[0], res.x[1], res.x[2])), color = 'r', label = 'fit')
                plt.plot(R, np.log10(sersic_np(R, x0[0], x0[1], x0[2])), color = 'orange', label = 'init')
                plt.legend()
                plt.title(f"{res.success} n {res.x[0]:0.3f} Rs {res.x[1]:0.3e} I0 {res.x[2]:0.3e}")
                plt.savefig(f"{self.name}_coma_test.jpg")
                plt.close()
                for i, param in enumerate([f"n_{r}", f"Rs_{r}", f"I0_{r}"]):
                    self[param].set_value(res.x[i], override_locked = (self[param].value is None))
                if self[f"Rs_{r}"].uncertainty is None:
                    self[f"Rs_{r}"].set_uncertainty(0.02 * self[f"Rs_{r}"].value.detach().item(), override_locked = True)
                if self[f"I0_{r}"].uncertainty is None:
                    self[f"I0_{r}"].set_uncertainty(0.02 * np.abs(self[f"I0_{r}"].value.detach().item()), override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        return sersic_torch(R, self[f"n_{i}"].value, self[f"Rs_{i}"].value, self[f"I0_{i}"].value * sample_image.pixelscale**2)
        
