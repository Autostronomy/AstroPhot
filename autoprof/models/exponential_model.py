from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .star_model_object import Star_Model
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .ray_model import Ray_Galaxy
from .edgeon_model import EdgeOn_Model
import torch
import numpy as np
from scipy.stats import iqr
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import exponential_torch, exponential_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import matplotlib.pyplot as plt
from scipy.optimize import minimize

__all__ = ["Exponential_Galaxy", "Exponential_Star", "Exponential_SuperEllipse", "Exponential_SuperEllipse_Warp", "Exponential_Warp", "Exponential_Ray", "Exponential_Exponential_EdgeOn", "Exponential_Sech2_EdgeOn"]

class Exponential_Galaxy(Galaxy_Model):
    """basic galaxy model with a exponential profile for the radial light
    profile.

    """
    model_type = f"exponential {Galaxy_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Galaxy_Model.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize

class Exponential_Star(Star_Model):
    """basic galaxy model with a exponential profile for the radial light
    profile.

    """
    model_type = f"exponential {Star_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Star_Model.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(self.radius_metric(X, Y), image)
    
class Exponential_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a exponential profile for the radial
    light profile.

    """
    model_type = f"exponential {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = SuperEllipse_Galaxy.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize

class Exponential_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a exponential profile for the
    radial light profile.

    """
    model_type = f"exponential {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = SuperEllipse_Warp.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize
    
class Exponential_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with an
    expoential profile for the radial light profile.

    """
    model_type = f"exponential {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = FourierEllipse_Galaxy.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize

class Exponential_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a exponential
    profile for the radial light profile.

    """
    model_type = f"exponential {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = FourierEllipse_Warp.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize
    
class Exponential_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a exponential profile for the
    radial light model.

    """
    model_type = f"exponential {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Warp_Galaxy.parameter_order + ("Re", "Ie")

    from ._shared_methods import exponential_radial_model as radial_model
    from ._shared_methods import exponential_initialize as initialize

class Exponential_Ray(Ray_Galaxy):
    """ray galaxy model with a exponential profile for the
    radial light model.

    """
    model_type = f"exponential {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Ray_Galaxy.parameter_order + ("Re", "Ie")

    def initialize(self):
        super(self.__class__, self).initialize()
        if all((self["Ie"].value is not None, self["Re"].value is not None)):
            return
        with torch.no_grad():
            # Get the sub-image area corresponding to the model image
            target_area = self.target[self.fit_window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(
                self["center"].value[0].detach().cpu().item(),
                self["center"].value[1].detach().cpu().item(), target_area
            )
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().cpu().item(), q = self["q"].value.detach().cpu().item(),
                n_isophotes = 15,
                more = True,
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
            for r in range(self.rays):
                flux = []
                for iso in iso_info:
                    modangles = (iso["angles"] - (self["PA"].value.detach().cpu().item() + r*np.pi/self.rays)) % np.pi
                    flux.append(np.median(iso["isovals"][np.logical_or(modangles < (0.5*np.pi/self.rays), modangles >= (np.pi*(1 - 0.5/self.rays)))]) / self.target.pixelscale**2)
                flux = np.array(flux)
                if np.sum(flux < 0) >= 1:
                    flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
                x0 = [
                    R[4] if self["Re"].value is None else self["Re"].value.detach().cpu().numpy()[r],
                    flux[4],
                ]
                res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(exponential_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
                self["Re"].set_value(res.x[1], override_locked = (self["Re"].value is None), index = r)
                self["Ie"].set_value(np.log10(res.x[2]), override_locked = (self["Ie"].value is None), index = r)
            if self["Re"].uncertainty is None:
                self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().numpy(), override_locked = True)
            if self["Ie"].uncertainty is None:
                self["Ie"].set_uncertainty(0.02 * len(self["Ie"].value), override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        return exponential_torch(R, self["Re"].value[i], (10**self["Ie"].value[i]) * sample_image.pixelscale**2)
        
class Exponential_Exponential_EdgeOn(EdgeOn_Model):
    """model for an edge-on galaxy with a exponential profile for the radial light
    profile and for the vertical light profile.

    """
    model_type = f"expexp {EdgeOn_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "Rr": {"units": "arcsec", "limits": (0,None)},
        "Rz": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Galaxy_Model.parameter_order + ("R0", "Rr", "Rz")

    def brightness_model(R, h, image):
        if sample_image is None:
            sample_image = self.model_image
        return self["I0"] * torch.exp(- R / self["Rr"]) * torch.exp(- h / self["Rz"])

class Exponential_Sech2_EdgeOn(EdgeOn_Model):
    """model for an edge-on galaxy with a exponential profile for the radial light
    profile and a sech^2 profile for the vertical component.

    """
    model_type = f"expsech2 {EdgeOn_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "Rr": {"units": "arcsec", "limits": (0,None)},
        "hz": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_order = Galaxy_Model.parameter_order + ("R0", "Rr", "hz")

    def brightness_model(R, h, image):
        if sample_image is None:
            sample_image = self.model_image
        return self["I0"] * torch.exp(- R / self["Rr"]) * torch.sech(- h / self["hz"])**2
