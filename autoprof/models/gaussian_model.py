from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .superellipse_model import SuperEllipse_Galaxy, SuperEllipse_Warp
from .foureirellipse_model import FourierEllipse_Galaxy, FourierEllipse_Warp
from .ray_model import Ray_Galaxy
from .star_model_object import Star_Model
from ..utils.parametric_profiles import gaussian_torch, gaussian_np
import torch
import numpy as np

__all__ = ["Gaussian_Galaxy", "Gaussian_SuperEllipse", "Gaussian_SuperEllipse_Warp", "Gaussian_FourierEllipse", "Gaussian_FourierEllipse_Warp", "Gaussian_Warp", "Gaussian_Star"]

class Gaussian_Galaxy(Galaxy_Model):
    """Basic galaxy model with Gaussian as the radial light profile. The
    gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total
    
    """
    model_type = f"gaussian {Galaxy_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_SuperEllipse(SuperEllipse_Galaxy):
    """Super ellipse galaxy model with Gaussian as the radial light
    profile.The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {SuperEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = SuperEllipse_Galaxy._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_SuperEllipse_Warp(SuperEllipse_Warp):
    """super ellipse warp galaxy model with a gaussian profile for the
    radial light profile. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {SuperEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = SuperEllipse_Warp._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize
    
class Gaussian_FourierEllipse(FourierEllipse_Galaxy):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile. The gaussian radial profile
    is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {FourierEllipse_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = FourierEllipse_Galaxy._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_FourierEllipse_Warp(FourierEllipse_Warp):
    """fourier mode perturbations to ellipse galaxy model with a gaussian
    profile for the radial light profile. The gaussian radial profile
    is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {FourierEllipse_Warp.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = FourierEllipse_Warp._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize

class Gaussian_Warp(Warp_Galaxy):
    """Coordinate warped galaxy model with Gaussian as the radial light
    profile. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {Warp_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("sigma", "flux")

    from ._shared_methods import gaussian_radial_model as radial_model
    from ._shared_methods import gaussian_initialize as initialize


class Gaussian_Star(Star_Model):
    """Basica star model with a Gaussian as the radial light profile. The
    gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {Star_Model.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = Star_Model._parameter_order + ("sigma", "flux")

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if self["sigma"].value is not None and self["flux"].value is not None:
            return
        target_area = target[self.window]
        if self["sigma"].value is None:
            self["sigma"].set_value(1, override_locked = True)
        if self["sigma"].uncertainty is None:
            self["sigma"].set_uncertainty(1e-2, override_locked = True)
        if self["flux"].value is None:
            self["flux"].set_value(np.sum(target_area.data.detach().cpu().numpy()), override_locked = True)
        if self["flux"].uncertainty is None:
            self["flux"].set_uncertainty(self["flux"].value.detach().cpu().numpy() * 1e-2, override_locked = True)
    from ._shared_methods import gaussian_radial_model as radial_model

    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        return self.radial_model(torch.sqrt(X**2 + Y**2), image)
        
class Gaussian_Ray(Ray_Galaxy):
    """ray galaxy model with a gaussian profile for the radial light
    model. The gaussian radial profile is defined as:

    I(R) = F * exp(-0.5 R^2/S^2) / sqrt(2pi*S^2)

    where I(R) is the prightness as a function of semi-major axis
    length, F is the total flux in the model, R is the semi-major
    axis, and S is the standard deviation.

    Parameters:
        sigma: standard deviation of the gaussian profile, must be a positive value
        flux: the total flux in the gaussian model, represented as the log of the total

    """
    model_type = f"gaussian {Ray_Galaxy.model_type}"
    parameter_specs = {
        "sigma": {"units": "arcsec", "limits": (0,None)},
        "flux": {"units": "log10(flux)", "limits": (0,None)},
    }
    _parameter_order = Ray_Galaxy._parameter_order + ("sigma", "flux")

    @torch.no_grad()
    def initialize(self, target = None):
        super(self.__class__, self).initialize()
        if self["sigma"].value is not None and self["flux"].value is not None:
            return
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
        was_none = [False, False]
        for i, p in enumerate(["sigma", "flux"]):
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
                R[4] if self["sigma"].value is None else self["sigma"].value.detach().cpu().numpy()[r],
                flux[4],
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(gaussian_np(R, x[0], x[1])))**2), x0 = x0, method = "SLSQP", bounds = ((R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
            if was_none[0]:
                self["sigma"].set_value(res.x[0], override_locked = True, index = r)
            if was_none[1]:
                self["flux"].set_value(np.log10(res.x[1]), override_locked = True, index = r)
        if self["sigma"].uncertainty is None:
            self["sigma"].set_uncertainty(0.02 * self["sigma"].value.detach().cpu().numpy(), override_locked = True)
        if self["flux"].uncertainty is None:
            self["flux"].set_uncertainty(0.02 * len(self["flux"].value), override_locked = True)
    
    def iradial_model(self, i, R, sample_image = None):
        if sample_image is None:
            sample_image = self.target
        return gaussian_torch(R, self["sigma"].value[i], (10**self["flux"].value[i])*sample_image.pixelscale**2)
    
