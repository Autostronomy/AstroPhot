from .galaxy_model_object import Galaxy_Model
from autoprof.utils.interpolate import cubic_spline_torch
import numpy as np
import torch
from autoprof.utils.conversions.coordinates import Axis_Ratio_Cartesian
from scipy.stats import iqr, binned_statistic, binned_statistic_2d

__all__ = ["Warp_Galaxy"]

class Warp_Galaxy(Galaxy_Model):
    """Galaxy model which includes radially varrying PA and q
    profiles. This works by warping the cooridnates using the same
    transform for a global PA/q except applied to each pixel
    individually. In the limit that PA and q are a constant, this
    recovers a basic galaxy model with global PA/q. However, a linear
    PA profile will give a spiral appearance, variations of PA/q
    profiles can create complex galaxy models.

    """
    model_type = " ".join(("warp", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0.05,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    parameter_order = Galaxy_Model.parameter_order + ("q(R)", "PA(R)")

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)

    def initialize(self):
        super().initialize()
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        if self["PA(R)"].value is None:
            self["PA(R)"].set_value(np.ones(len(self.profR))*self["PA"].value.detach().cpu().item(), override_locked = True)
            
        if self["q(R)"].value is None:
            self["q(R)"].set_value(np.ones(len(self.profR))*0.9, override_locked = True)
            
    def set_fit_window(self, window):
        super().set_fit_window(window)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < min(self.fit_window.shape/2):
                self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
            self.profR.pop()
            self.profR = torch.tensor(self.profR)

    def transform_coordinates(self, X, Y):
        X, Y = super().transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        PA = cubic_spline_torch(self.profR, self["PA(R)"].value, R.view(-1)).view(*R.shape)
        q = cubic_spline_torch(self.profR, self["q(R)"].value, R.view(-1)).view(*R.shape)
        return Axis_Ratio_Cartesian(q, X, Y, PA, inv_scale = True) # fixme check inv_scale
