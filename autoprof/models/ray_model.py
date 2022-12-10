from .galaxy_model_object import Galaxy_Model
from autoprof.utils.interpolate import cubic_spline_torch
from .parameter_object import Parameter
import numpy as np
import torch
from autoprof.utils.conversions.coordinates import Axis_Ratio_Cartesian

__all__ = ["Ray_Galaxy"]

class Ray_Galaxy(Galaxy_Model):
    """Variant of a galaxy model which defines multiple radial models
    seprarately along some number of rays projected from the galaxy
    center. These rays smoothly transition from one to another along
    angles theta. The ray transition uses a cosine smoothing function
    which depends on the number of rays, for example with two rays the
    brightness would be:

    I(R,theta) = I1(R)*cos(theta % pi) + I2(R)*cos((theta + pi/2) % pi)

    Where I(R,theta) is the brightness function in polar coordinates,
    R is the semi-major axis, theta is the polar angle (defined after
    galaxy axis ratio is applied), I1(R) is the first brightness
    profile, % is the modulo operator, and I2 is the second brightness
    profile. The ray model defines no extra parameters, though now
    every model parameter related to the brightness profile gains an
    extra dimension for the ray number. Also a new input can be given
    when instantiating the ray model: "rays" which is an integer for
    the number of rays.

    """
    model_type = f"ray {Galaxy_Model.model_type}"
    special_kwargs = Galaxy_Model.special_kwargs + ["rays"]
    def __init__(self, *args, **kwargs):
        self.symmetric_rays = True
        super().__init__(*args, **kwargs)
        self.rays = kwargs.get("rays", 2)
        
    def angular_metric(self, X, Y):
        return torch.atan2(Y, X)
    
    def polar_model(self, R, T, image):
        model = torch.zeros(R.shape)
        if self.rays % 2 == 0 and self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r*np.pi/self.rays)) % np.pi
                indices = torch.logical_or(angles < (np.pi/self.rays), angles >= (np.pi*(1 - 1/self.rays)))
                weight = (torch.cos(angles[indices] * self.rays) + 1)/2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        elif self.rays % 2 == 1 and self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r*np.pi/self.rays)) % (2*np.pi)
                indices = torch.logical_or(angles < (np.pi/self.rays), angles >= (np.pi*(2 - 1/self.rays)))
                weight = (torch.cos(angles[indices] * self.rays) + 1)/2
                model[indices] += weight * self.iradial_model(r, R[indices], image) 
                angles = (T - (np.pi + r*np.pi/self.rays)) % (2*np.pi)
                indices = torch.logical_or(angles < (np.pi/self.rays), angles >= (np.pi*(2 - 1/self.rays)))
                weight = (torch.cos(angles[indices] * self.rays) + 1)/2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        elif self.rays % 2 == 0 and not self.symmetric_rays:
            for r in range(self.rays):
                angles = (T - (r*2*np.pi/self.rays)) % (2*np.pi)
                indices = torch.logical_or(angles < (2*np.pi/self.rays), angles >= (2*np.pi*(1 - 1/self.rays)))
                weight = (torch.cos(angles[indices] * self.rays) + 1)/2
                model[indices] += weight * self.iradial_model(r, R[indices], image)
        else:
            for r in range(self.rays):
                angles = (T - (r*2*np.pi/self.rays)) % (2*np.pi)
                indices = torch.logical_or(angles < (2*np.pi/self.rays), angles >= (np.pi*(2 - 1/self.rays)))
                weight = (torch.cos(angles[indices] * self.rays) + 1)/2
                model[indices] += weight * self.iradial_model(r, R[indices], image) 
        return model
    
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        XX, YY = self.transform_coordinates(X, Y)
        
        return self.polar_model(self.radius_metric(XX, YY), self.angular_metric(XX, YY), image)

#class SingleRay_Galaxy(Galaxy_Model):

