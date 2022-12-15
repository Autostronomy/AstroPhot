from .galaxy_model_object import Galaxy_Model
from .parameter_object import Parameter
from ..utils.interpolate import cubic_spline_torch
from ..utils.conversions.coordinates import Axis_Ratio_Cartesian
import numpy as np
import torch

__all__ = ["Wedge_Galaxy"]

class Wedge_Galaxy(Galaxy_Model):
    """Variant of the ray model where no smooth transition is performed
    between regions as a function of theta, instead there is a sharp
    trnasition boundary. This may be desireable as it cleanly
    separates where the pixel information is going. Due to the sharp
    transition though, it may cause unusual behaviour when fitting. If
    problems occur, try fitting a ray model first then fix the center,
    PA, and q and then fit the wedge model. Essentially this breaks
    down the structure fitting and the light profile fitting into two
    steps. The wedge model, like the ray model, defines no extra
    parameters, however a new option can be supplied on instantiation
    of the wedge model which is "wedges" or the number of wedges in
    the model.

    """

    model_type = f"wedge {Galaxy_Model.model_type}"
    special_kwargs = Galaxy_Model.special_kwargs + ["wedges"]
    def __init__(self, *args, **kwargs):
        self.symmetric_wedges = True
        super().__init__(*args, **kwargs)
        self.wedges = kwargs.get("wedges", 2)
        
    def angular_metric(self, X, Y):
        return torch.atan2(Y, X)
    
    def polar_model(self, R, T, image):
        model = torch.zeros(R.shape)
        if self.wedges % 2 == 0 and self.symmetric_wedges:
            for w in range(self.wedges):
                angles = (T - (w*np.pi/self.wedges)) % np.pi
                indices = torch.logical_or(angles < (np.pi/(2*self.wedges)), angles >= (np.pi*(1 - 1/(2*self.wedges))))
                model[indices] += self.iradial_model(w, R[indices], image)
        elif self.wedges % 2 == 1 and self.symmetric_wedges:
            for w in range(self.wedges):
                angles = (T - (w*np.pi/self.wedges)) % (2*np.pi)
                indices = torch.logical_or(angles < (np.pi/(2*self.wedges)), angles >= (np.pi*(2 - 1/(2*self.wedges))))
                model[indices] += self.iradial_model(w, R[indices], image) 
                angles = (T - (np.pi + w*np.pi/self.wedges)) % (2*np.pi)
                indices = torch.logical_or(angles < (np.pi/(2*self.wedges)), angles >= (np.pi*(2 - 1/(2*self.wedges))))
                model[indices] += self.iradial_model(w, R[indices], image)
        else:
            for w in range(self.wedges):
                angles = (T - (w*2*np.pi/self.wedges)) % (2*np.pi)
                indices = torch.logical_or(angles < (np.pi/self.wedges), angles >= (np.pi*(2 - 1/self.wedges)))
                model[indices] += self.iradial_model(w, R[indices], image)
        return model
    
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        XX, YY = self.transform_coordinates(X, Y)
        
        return self.polar_model(self.radius_metric(XX, YY), self.angular_metric(XX, YY), image)
