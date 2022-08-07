from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
from scipy.stats import iqr
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Sersic_Galaxy(Galaxy_Model):

    model_type = " ".join(("sersic", Galaxy_Model.model_type))
    parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_qualities = {
        "I0": {"form": "value"},
        "n": {"form": "value"},
        "Rs": {"form": "value"},
    }

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if any((self["n"].value is None, self["I0"].value is None, self["Rs"].value is None)):
            # Get the sub-image area corresponding to the model image
            target_area = target[self.window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(self["center"][0].value, self["center"][1].value, target_area)
            iso_info = isophotes(
                target_area.data,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value, q = self["q"].value,
                n_isophotes = 15
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * target.pixelscale
            flux = np.array(list(iso["flux"] for iso in iso_info)) / target.pixelscale**2
            if np.sum(flux < 0) > 5:
                flux -= np.min(flux)*1.01
            CHOOSE = flux > 0
            R = R[CHOOSE]
            flux = flux[CHOOSE]
            x0 = [
                2. if self["n"].value is None else self["n"].value,
                R[4] if self["Rs"].value is None else self["Rs"].value,
                flux[0] if self["I0"].value is None else self["I0"].value,
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic(R, np.abs(x[0]), np.abs(x[1]), np.abs(x[2]))))**2), x0 = x0, method = 'Nelder-Mead')
            # plt.scatter(R,flux)
            # plt.plot(R, sersic(R, x0[0], x0[1], x0[2]))
            # plt.plot(R, sersic(R, res.x[0], res.x[1], res.x[2]))
            # plt.savefig(f"deleteme_sersic_{self.name}.jpg")
            # plt.close()
            for i, param in enumerate(["n", "Rs", "I0"]):
                self[param].set_value(np.abs(res.x[i]), override_fixed = (self[param].value is None))
        if self["Rs"].uncertainty is None:
            self["Rs"].set_uncertainty(0.02 * self["Rs"].value, override_fixed = True)
        if self["I0"].uncertainty is None:
            self["I0"].set_uncertainty(0.02 * np.abs(self["I0"].value), override_fixed = True)

    def radial_model(self, R, sample_image = None):
        
        if sample_image is None:
            sample_image = self.model_image        
        return sersic(R, self["n"].value, self["Rs"].value, self["I0"].value * sample_image.pixelscale**2)

    
class Sersic_Warp(Warp_Galaxy):

    model_type = " ".join(("sersic", Warp_Galaxy.model_type))
    parameter_specs = {
        "I0": {"units": "flux/arcsec^2"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }
    parameter_qualities = {
        "I0": {"form": "value"},
        "n": {"form": "value"},
        "Rs": {"form": "value"},
    }

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if any((self["n"].value is None, self["I0"].value is None, self["Rs"].value is None)):
            # Get the sub-image area corresponding to the model image
            target_area = target[self.window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(self["center"][0].value, self["center"][1].value, target_area)
            iso_info = isophotes(
                target_area.data - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value, q = self["q"].value,
                n_isophotes = 15
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * target.pixelscale
            flux = np.array(list(iso["flux"] for iso in iso_info)) / target.pixelscale**2
            if np.sum(flux < 0) > 1:
                flux -= np.min(flux)
            x0 = [
                2. if self["n"].value is None else self["n"].value,
                R[5] if self["Rs"].value is None else self["Rs"].value,
                flux[0] if self["I0"].value is None else self["I0"].value,
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic(R, x[0], x[1], x[2])))**2), x0 = x0, method = 'Nelder-Mead')
            for i, param in enumerate(["n", "Rs", "I0"]):
                self[param].set_value(res.x[i], override_fixed = (self[param].value is None))
        if self["Rs"].uncertainty is None:
            self["Rs"].set_uncertainty(0.02 * self["Rs"].value, override_fixed = True)
        if self["I0"].uncertainty is None:
            self["I0"].set_uncertainty(0.02 * np.abs(self["I0"].value), override_fixed = True)

    def radial_model(self, R, sample_image = None):
        
        if sample_image is None:
            sample_image = self.model_image        
        return sersic(R, self["n"].value, self["Rs"].value, self["I0"].value * sample_image.pixelscale**2)

