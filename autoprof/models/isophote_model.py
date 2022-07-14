from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .parameter_object import Parameter_Array
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import binned_statistic, iqr

class Isophotal_Galaxy(Galaxy_Model):

    model_type = " ".join(("isophotal", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    parameter_qualities = {
        "q(R)": {"form": "array", "loss": "band fft2", "regularize": "self", "regularize scale": 1.},
        "PA(R)": {"form": "array", "loss": "band fft2", "regularize": "const", "regularize scale": 0.2},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
        
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.window.shape/2)**2)):
                self.profR.append(max(1,self.profR[-1]*1.2))
            self.profR.pop()                
            self.profR = np.array(self.profR)
        
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        
        if self["PA(R)"].value is None:
            self["PA(R)"].set_value(np.ones(len(self.profR))*self["PA"].value, override_fixed = True)
            
        if self["q(R)"].value is None:
            self["q(R)"].set_value(np.ones(len(self.profR))*0.9, override_fixed = True)

    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(data)

        icenter = coord_to_index(self["center"][0].value, self["center"][1].value, data.residual_image)
        
        samples = isophotes(data.residual_image.data, icenter, pa = self["PA(R)"].get_value(), q = self["q(R)"].get_value(), R = self.profR / data.residual_image.pixelscale)

        
        
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value() * sample_image.pixelscale**2, ext = "zeros", s = 0)
        return I(R)
