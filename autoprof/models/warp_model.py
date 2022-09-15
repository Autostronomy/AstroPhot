from .galaxy_model_object import Galaxy_Model
from .parameter_object import Parameter_Array
import numpy as np
from scipy.interpolate import UnivariateSpline
from autoprof.utils.initialize import isophotes
from autoprof.utils.interpolate import nearest_neighbor
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.fftpack import fft, ifft
from scipy.stats import iqr, binned_statistic, binned_statistic_2d
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize

class Warp_Galaxy(Galaxy_Model):

    model_type = " ".join(("warp", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    parameter_qualities = {
        "q(R)": {"form": "array", "regularize": "self", "regularize scale": 1.},
        "PA(R)": {"form": "array", "regularize": "const", "regularize scale": 0.2},
    }

    fft_start = 10

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)

    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["PA(R)"].value is not None:
            self["PA(R)"].set_value(self["PA(R)"].value * np.pi / 180, override_fixed = True)

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        if self["PA(R)"].value is None:
            self["PA(R)"].set_value(np.ones(len(self.profR))*self["PA"].value, override_fixed = True)
            
        if self["q(R)"].value is None:
            self["q(R)"].set_value(np.ones(len(self.profR))*0.9, override_fixed = True)
            
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < max(self.window.shape/2):
                self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
            self.profR.pop()
            self.profR = np.array(self.profR)

    def transform_coordinates(self, X, Y, R = None, transmit = True):
        if transmit:
            X, Y = super().transform_coordinates(X, Y)

        if R is None:
            R = self.radius_metric(X, Y)
        PA = UnivariateSpline(self.profR, np.unwrap(self["PA(R)"].value*2)/2, ext = "const", s = 0)
        q = UnivariateSpline(self.profR, self["q(R)"].value, ext = "const", s = 0)
        
        return Axis_Ratio_Cartesian(q(R), X, Y, PA(R), inv_scale = True)

    def regularize_loss(self):

        params = {"q(R)": self["q(R)"], "PA(R)": self["PA(R)"]}
        regularization = np.ones(len(self.profR))
        for P in params:
            if not isinstance(params[P], Parameter_Array):
                continue
            vals = params[P].value
            if params[P].cyclic:
                period_factor = 2*np.pi / (params[P].limits[1] - params[P].limits[0])
                vals = np.unwrap(vals * period_factor) / period_factor
            if self.parameter_qualities[P]["regularize"] == "const":
                reg_scale = np.ones(len(self.profR))
            elif self.parameter_qualities[P]["regularize"] == "self":
                reg_scale = vals
            reg_scale *= self.parameter_qualities[P]["regularize scale"]
            reg = [2*np.abs(vals[1] - vals[0]) / reg_scale[0]]
            for i in range(1, len(vals) - 1):
                reg.append((np.abs(vals[i] - vals[i-1]) + np.abs(vals[i] - vals[i+1])) / reg_scale[i])
            reg.append(2*np.abs(vals[-2] - vals[-1]) / reg_scale[-1])
            regularization += 0.01*np.array(reg)

        return regularization
