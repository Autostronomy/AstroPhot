from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
import torch
import numpy as np

__all__ = ["FourierEllipse_Galaxy", "FourierEllipse_Warp"]

class FourierEllipse_Galaxy(Galaxy_Model):
    """Expanded galaxy model which includes a Fourier transformation in
    its radius metric. This allows for the expression of arbitrarily
    complex isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation.

    """
    model_type = f"fourier {Galaxy_Model.model_type}"
    parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "limits": (0, 2*np.pi), "cyclic": True}
    }
    parameter_order = Galaxy_Model.parameter_order + ("am", "phim")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = torch.tensor(kwargs.get("modes", (1,3,4)))
    
    def angular_metric(self, X, Y):
        return torch.atan2(Y, X)
    
    def radius_metric(self, X, Y):
        R = super().radius_metric(X, Y)
        theta = self.angular_metric(X, Y)
        return R * torch.exp(torch.sum(self["am"].value.view(len(self.modes), -1)*torch.cos(self.modes.view(len(self.modes), -1)*theta.view(-1) + self["phim"].value.view(len(self.modes), -1)), 0).view(theta.shape))

    def initialize(self):
        super().initialize()

        self["am"].set_value(torch.zeros(len(self.modes)), override_locked = self["am"].value is None)
        self["am"].set_uncertainty(torch.tensor(0.05*np.ones(len(self.modes))), override_locked = self["am"].uncertainty is None)
        self["phim"].set_value(torch.zeros(len(self.modes)), override_locked = self["phim"].value is None)
        self["phim"].set_uncertainty(torch.tensor((5*np.pi/180)*np.ones(len(self.modes))), override_locked = self["phim"].uncertainty is None)


class FourierEllipse_Warp(Warp_Galaxy):
    """Expanded warp galaxy model which includes a Fourier transformation in
    its radius metric. This allows for the expression of arbitrarily
    complex isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation.

    """
    model_type = f"fourier {Warp_Galaxy.model_type}"
    parameter_specs = {
        "am": {"units": "none"},
        "phim": {"units": "radians", "limits": (0, 2*np.pi), "cyclic": True}
    }
    parameter_order = Warp_Galaxy.parameter_order + ("am", "phim")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modes = torch.tensor(kwargs.get("modes", (1,3,4)))
    
    def angular_metric(self, X, Y):
        return torch.atan2(Y, X)
    
    def radius_metric(self, X, Y):
        R = super().radius_metric(X, Y)
        theta = self.angular_metric(X, Y)
        return R * torch.exp(torch.sum(self["am"].value.view(len(self.modes), -1)*torch.cos(self.modes.view(len(self.modes), -1)*theta.view(-1) + self["phim"].value.view(len(self.modes), -1)), 0).view(theta.shape))

    def initialize(self):
        super().initialize()

        self["am"].set_value(torch.zeros(len(self.modes)), override_locked = self["am"].value is None)
        self["am"].set_uncertainty(torch.tensor(0.05*np.ones(len(self.modes))), override_locked = self["am"].uncertainty is None)
        self["phim"].set_value(torch.zeros(len(self.modes)), override_locked = self["phim"].value is None)
        self["phim"].set_uncertainty(torch.tensor((5*np.pi/180)*np.ones(len(self.modes))), override_locked = self["phim"].uncertainty is None)

