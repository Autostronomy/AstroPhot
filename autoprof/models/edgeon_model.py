from .model_object import Component_Model
from ..utils.initialize import isophotes
from ..utils.angle_operations import Angle_Average
from ..utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr
import torch
import numpy as np

__all__ = ["Edgeon_Model"]

class Edgeon_Model(Component_Model):
    """General Edge-On galaxy model to be subclassed for any specific
    representation such as radial light profile or the structure of
    the galaxy on the sky. Defines an edgeon galaxy as an object with
    a position angle, no inclination information is included.

    """
    model_type = f"edgeon {Component_Model.model_type}"
    parameter_specs = {
        "PA": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
    }
    _parameter_order = Component_Model._parameter_order + ("PA", )
    useable = False

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target        
        super().initialize(target)
        if self["PA"].value is not None:
            return
        target_area = target[self.window]
        edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng = (16,84))/2
        icenter = coord_to_index(self["center"].value[0], self["center"].value[1], target_area)
        iso_info = isophotes(
            target_area.data.detach().cpu().numpy() - edge_average,
            (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
            threshold = 3*edge_scatter,
            pa = 0., q = 1., n_isophotes = 15
        )
        self["PA"].set_value((-Angle_Average(list(iso["phase2"] for iso in iso_info[-int(len(iso_info)/3):]))/2) % np.pi, override_locked = True)

    def transform_coordinates(self, X, Y):
        return Rotate_Cartesian(-self["PA"].value, X, Y)
        
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        XX, YY = self.transform_coordinates(X, Y)
        
        return self.brightness_model(torch.abs(XX), torch.abs(YY), image)

class Edgeon_Sech(Edgeon_Model):
    """An edgeon profile where the vertical distribution is a sech^2
    profile, subclasses define the radial profile.

    """
    
    model_type = f"sech2 {Edgeon_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "log10(flux/arcsec^2)"},
        "hs": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = Edgeon_Model._parameter_order + ("I0", "hs")
    useable = False

    def brightness_model(self, X, Y, image):
        return (image.pixelscale**2)*(10**self["I0"].value) * self.radial_model(X) / (torch.cosh(Y / self["hs"].value)**2)

class Edgeon_Isothermal(Edgeon_Sech):
    """A self-gravitating locally-isothermal edgeon disk. This comes from
    van der Kruit & Searle 1981.

    """
    model_type = f"isothermal {Edgeon_Sech.model_type}"
    parameter_specs = {
        "rs": {"units": "arcsec", "limits": (0,None)},
    }
    _parameter_order = Edgeon_Sech._parameter_order + ("rs",)
    useable = True

    def radial_model(self, R):
        Rscaled = torch.abs(R / self["rs"].value)
        return Rscaled * torch.exp(-Rscaled) * torch.special.scaled_modified_bessel_k1(Rscaled)
    
