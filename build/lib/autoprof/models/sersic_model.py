from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .ray_model import Ray_Galaxy
from .star_model_object import Star_Model
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .edgeon_model import EdgeOn_Model
import torch
import numpy as np
from scipy.stats import iqr
from ..utils.initialize import isophotes
from ..utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np
from ..utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
from scipy.optimize import minimize

__all__ = [
    "Sersic_Galaxy", "Sersic_Star", "Sersic_Warp",
    "Sersic_SuperEllipse", "Sersic_FourierEllipse", "Sersic_Ray",
    "Sersic_SuperEllipse_Warp", "Sersic_FourierEllipse_Warp"
] # , "Sersic_Sersic_EdgeOn", "Sersic_Sech2_EdgeOn"

class Sersic_Galaxy(Galaxy_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius
    
    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
        "Ie": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Star(Star_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {Star_Model.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = Star_Model._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(self.radius_metric(X, Y), image)
    
class Sersic_SuperEllipse(SuperEllipse_Galaxy):
    """super ellipse galaxy model with a sersic profile for the radial
    light profile. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a sersic profile for the
    radial light profile. The functional form of the Sersic profile is
    defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a sersic
    profile for the radial light profile. The functional form of the
    Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a sersic
    profile for the radial light profile. The functional form of the
    Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    
class Sersic_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a sersic profile for the radial
    light model. The functional form of the Sersic profile is defined
    as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {Warp_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("n", "Re", "Ie")

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize

class Sersic_Ray(Ray_Galaxy):
    """ray galaxy model with a sersic profile for the radial light
    model. The functional form of the Sersic profile is defined as:

    I(R) = Ie * exp(- bn((R/Re)^(1/n) - 1))

    where I(R) is the brightness profile as a function of semi-major
    axis, R is the semi-major axis length, Ie is the brightness as the
    half light radius, bn is a function of n and is not involved in
    the fit, Re is the half light radius, and n is the sersic index
    which controls the shape of the profile.

    Parameters:
        n: Sersic index which controls the shape of the brightness profile
        Ie: brightness at the half light radius, represented as the log of the brightness divided by pixel scale squared.
        Re: half light radius

    """
    model_type = f"sersic {Ray_Galaxy.model_type}"
    parameter_specs = {
        "Ie": {"units": "log10(flux/arcsec^2)"},
        "n": {"units": "none", "limits": (0.36,8), "uncertainty": 0.05},
        "Re": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("n", "Re", "Ie")

    @torch.no_grad()
    def initialize(self, target = None):
        super(self.__class__, self).initialize()
        if all((self["n"].value is not None, self["Ie"].value is not None, self["Re"].value is not None)):
            return# fixme need to initialize n, Ie, Re as tensors
        # Get the sub-image area corresponding to the model image
        target_area = self.target[self.window]
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
        was_none = [False, False, False]
        for i, p in enumerate(["n", "Re", "Ie"]):
            if self[p].value is None:
                was_none[i] = True
                self[p].set_value(np.zeros(self.rays), override_locked = True)
        for r in range(self.rays):
            flux = []
            for iso in iso_info:
                modangles = (iso["angles"] - (self["PA"].value.detach().cpu().item() + r*np.pi/self.rays)) % np.pi
                flux.append(np.median(iso["isovals"][np.logical_or(modangles < (0.5*np.pi/self.rays), modangles >= (np.pi*(1 - 0.5/self.rays)))]) / self.target.pixelscale**2)
            flux = np.array(flux)
            if np.sum(flux < 0) >= 1:
                flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
            x0 = [
                2. if self["n"].value is None else self["n"].value.detach().cpu().numpy()[r],
                R[4] if self["Re"].value is None else self["Re"].value.detach().cpu().numpy()[r],
                flux[4],
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
            
            if was_none[0]:
                self["n"].set_value(res.x[0], override_locked = True, index = r)
            if was_none[1]:
                self["Re"].set_value(res.x[1], override_locked = True, index = r)
            if was_none[2]:
                self["Ie"].set_value(np.log10(res.x[2]), override_locked = True, index = r)
        if self["Re"].uncertainty is None:
            self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().numpy(), override_locked = True)
        if self["Ie"].uncertainty is None:
            self["Ie"].set_uncertainty(0.02 * len(self["Ie"].value), override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.target
        return sersic_torch(R, self[f"n"].value[i], self[f"Re"].value[i], (10**self[f"Ie"].value[i]) * sample_image.pixelscale**2)
        
# class Sersic_Sersic_EdgeOn(EdgeOn_Model):
#     """model for an edge-on galaxy with a exponential profile for the radial light
#     profile and for the vertical light profile.

#     """
#     model_type = f"sersicsersic {EdgeOn_Model.model_type}"
#     parameter_specs = {
#         "I0": {"units": "log10(flux/arcsec^2)"},
#         "Rr": {"units": "arcsec", "limits": (0,None)},
#         "Rz": {"units": "arcsec", "limits": (0,None)},
#     }
#     _parameter_order = Galaxy_Model._parameter_order + ("R0", "Rr", "Rz")

#     def brightness_model(R, h, image):
#         if sample_image is None:
#             sample_image = self.model_image
#         return self["I0"] * torch.exp(- R / self["Rr"]) * torch.exp(- h / self["Rz"])

# class Sersic_Sech2_EdgeOn(EdgeOn_Model):
#     """model for an edge-on galaxy with a exponential profile for the radial light
#     profile and a sech^2 profile for the vertical component.

#     """
#     model_type = f"sersicsech2 {EdgeOn_Model.model_type}"
#     parameter_specs = {
#         "I0": {"units": "log10(flux/arcsec^2)"},
#         "Rr": {"units": "arcsec", "limits": (0,None)},
#         "hz": {"units": "arcsec", "limits": (0,None)},
#     }
#     _parameter_order = Galaxy_Model._parameter_order + ("R0", "Rr", "hz")

#     def brightness_model(R, h, image):
#         if sample_image is None:
#             sample_image = self.model_image
#         return self["I0"] * torch.exp(- R / self["Rr"]) * torch.sech(- h / self["hz"])**2
