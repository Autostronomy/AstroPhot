from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
import torch
import numpy as np

__all__ = ["SuperEllipse_Galaxy", "SuperEllipse_Warp"]

class SuperEllipse_Galaxy(Galaxy_Model):
    """Expanded galaxy model which includes a superellipse transformation
    in its radius metric. This allows for the expression of "boxy" and
    "disky" isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation, especially
    for early-type galaxies. The functional form for this is:

    R = (|X|^C + |Y|^C)^(1/C)

    where R is the new distance metric, X Y are the coordiantes, and C
    is the coefficient for the superellipse. C can take on any value
    greater than zero where C = 2 is the standard distance metric, 0 <
    C < 2 creates disky or pointed perturbations to an ellipse, and C
    > 2 transforms an ellipse to be more boxy.

    Parameters:
        C0: superellipse distance metric parameter where C0 = C-2 so that a value of zero is now a standard ellipse.
    
    """
    model_type = f"superellipse {Galaxy_Model.model_type}"
    parameter_specs = {
        "C0": {"units": "C-2", "value": 0., "uncertainty": 1e-2, "limits": (-2, None)},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("C0",)
    
    def radius_metric(self, X, Y):
        return torch.pow(torch.pow(torch.abs(X)+1e-6, self["C0"].value + 2.) + torch.pow(torch.abs(Y)+1e-6, self["C0"].value + 2.), 1. / (self["C0"].value + 2.)) # epsilon added for numerical stability of gradient

class SuperEllipse_Warp(Warp_Galaxy):
    """Expanded warp model which includes a superellipse transformation
    in its radius metric. This allows for the expression of "boxy" and
    "disky" isophotes instead of pure ellipses. This is a common
    extension of the standard elliptical representation, especially
    for early-type galaxies. The functional form for this is:

    R = (|X|^C + |Y|^C)^(1/C)

    where R is the new distance metric, X Y are the coordiantes, and C
    is the coefficient for the superellipse. C can take on any value
    greater than zero where C = 2 is the standard distance metric, 0 <
    C < 2 creates disky or pointed perturbations to an ellipse, and C
    > 2 transforms an ellipse to be more boxy.

    Parameters:
        C0: superellipse distance metric parameter where C0 = C-2 so that a value of zero is now a standard ellipse.


    """
    model_type = f"superellipse {Warp_Galaxy.model_type}"
    parameter_specs = {
        "C0": {"units": "C-2", "value": 0., "uncertainty": 1e-2, "limits": (-2, None)},
    }
    _parameter_order = Warp_Galaxy._parameter_order + ("C0",)
    
    def radius_metric(self, X, Y):
        return torch.pow(torch.pow(torch.abs(X)+1e-6, self["C0"].value + 2.) + torch.pow(torch.abs(Y)+1e-6, self["C0"].value + 2.), 1. / (self["C0"].value + 2.)) # epsilon added for numerical stability of gradient

    
