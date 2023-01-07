from .galaxy_model_object import Galaxy_Model
from ..utils.interpolate import cubic_spline_torch
from ..utils.conversions.coordinates import Axis_Ratio_Cartesian, Rotate_Cartesian
import numpy as np
import torch
from scipy.stats import iqr, binned_statistic, binned_statistic_2d

__all__ = ["Warp_Galaxy"]

class Warp_Galaxy(Galaxy_Model):
    """Galaxy model which includes radially varrying PA and q
    profiles. This works by warping the cooridnates using the same
    transform for a global PA/q except applied to each pixel
    individually. In the limit that PA and q are a constant, this
    recovers a basic galaxy model with global PA/q. However, a linear
    PA profile will give a spiral appearance, variations of PA/q
    profiles can create complex galaxy models. The form of the
    coordinate transformation looks like:

    X, Y = meshgrid(image)
    R = sqrt(X^2 + Y^2)
    X', Y' = Rot(theta(R), X, Y)
    Y'' = Y' / q(R)

    where the definitions are the same as for a regular galaxy model,
    except now the theta is a function of radius R (before
    transformation) and the axis ratio q is also a function of radius
    (before the transformation).
    
    Parameters:
        q(R): Tensor of axis ratio values for axis ratio spline
        PA(R): Tensor of position angle values as input to the spline

    """
    model_type = f"warp {Galaxy_Model.model_type}"
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0.05,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    _parameter_order = Galaxy_Model._parameter_order + ("q(R)", "PA(R)")

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        
        # create the PA(R) and q(R) profile radii if needed
        for prof_param in ["PA(R)", "q(R)"]:
            if self[prof_param].prof is None:
                if self[prof_param].value is None: # from scratch
                    new_prof = [0,2*target.pixelscale]
                    while new_prof[-1] < torch.min(self.window.shape/2):
                        new_prof.append(new_prof[-1] + torch.max(2*target.pixelscale,new_prof[-1]*0.2))
                    new_prof.pop()
                    new_prof.pop()
                    new_prof.append(torch.sqrt(torch.sum((self.window.shape/2)**2)))
                    self[prof_param].set_profile(new_prof)
                else: # matching length of a provided profile
                    # create logarithmically spaced profile radii
                    new_prof = [0] + list(np.logspace(
                        np.log10(2*target.pixelscale),
                        np.log10(torch.max(self.window.shape/2).item()),
                        len(self[prof_param].value) - 1,
                    ))
                    # ensure no step is smaller than a pixelscale
                    for i in range(1,len(new_prof)):
                        if new_prof[i] - new_prof[i-1] < target.pixelscale.item():
                            new_prof[i] = new_prof[i-1] + target.pixelscale.item()
                    self[prof_param].set_profile(new_prof)
                    
                
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        if self["PA(R)"].value is None:
            self["PA(R)"].set_value(np.zeros(len(self["PA(R)"].prof)), override_locked = True)
            
        if self["q(R)"].value is None:
            self["q(R)"].set_value(np.ones(len(self["q(R)"].prof))*0.9, override_locked = True)
        
    def transform_coordinates(self, X, Y):
        X, Y = super().transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        PA = cubic_spline_torch(self["PA(R)"].prof, -self["PA(R)"].value, R.view(-1)).view(*R.shape)
        q = cubic_spline_torch(self["q(R)"].prof, self["q(R)"].value, R.view(-1)).view(*R.shape)
        X, Y = Rotate_Cartesian(PA, X, Y)
        return X, Y/q 
