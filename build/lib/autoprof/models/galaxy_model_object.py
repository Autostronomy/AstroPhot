from .model_object import BaseModel
from ..utils.initialize import isophotes
from ..utils.angle_operations import Angle_Average
from ..utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr
import torch
import numpy as np

__all__ = ["Galaxy_Model"]

class Galaxy_Model(BaseModel):
    """General galaxy model to be subclassed for any specific
    representation. Defines a galaxy as an object with a position
    angle and axis ratio, or effectively a tilted disk. Most
    subclassing models should simply define a radial model or update
    to the coordinate transform. The definition of the position angle and axis ratio used here is simply a scaling along the minor axis. The transformation can be written as:

    X, Y = meshgrid(image)
    X', Y' = Rot(theta, X, Y)
    Y'' = Y' / q

    where X Y are the coordinates of an image, X' Y' are the rotated
    coordinates, Rot is a rotation matrix by angle theta applied to the
    initial X Y coordinates, Y'' is the scaled semi-minor axis, and q
    is the axis ratio.

    Parameters:
        q: axis ratio to scale minor axis from the ratio of the minor/major axis b/a, this parameter is unitless, it is restricted to the range (0,1) 
        PA: position angle of the smei-major axis relative to the image positive x-axis in radians, it is a cyclic parameter in the range [0,pi)

    """
    model_type = f"galaxy {BaseModel.model_type}"
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0,1), "uncertainty": 0.03},
        "PA": {"units": "radians", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
    }
    _parameter_order = BaseModel._parameter_order + ("q", "PA")

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if not (self["PA"].value is None or self["q"].value is None):
            return
        target_area = target[self.window]
        edge = np.concatenate((
            target_area.data.detach().cpu().numpy()[:,0],
            target_area.data.detach().cpu().numpy()[:,-1],
            target_area.data.detach().cpu().numpy()[0,:],
            target_area.data.detach().cpu().numpy()[-1,:]
        ))
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng = (16,84))/2
        icenter = coord_to_index(self["center"].value[0], self["center"].value[1], target_area)
        if self["PA"].value is None:
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold = 3*edge_scatter,
                pa = 0., q = 1., n_isophotes = 15
            )
            self["PA"].set_value((-Angle_Average(list(iso["phase2"] for iso in iso_info[-int(len(iso_info)/3):]))/2) % np.pi, override_locked = True)
        if self["q"].value is None:
            q_samples = np.linspace(0.1,0.9,15)
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1].detach().cpu().item(), icenter[0].detach().cpu().item()),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().cpu().item(), q = q_samples,
            ) 
            self["q"].set_value(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))], override_locked = True)

    def radius_metric(self, X, Y):
        return torch.sqrt((torch.abs(X)+1e-8)**2 + (torch.abs(Y)+1e-8)**2) # epsilon added for numerical stability of gradient

    def transform_coordinates(self, X, Y):
        X, Y = Rotate_Cartesian(-self["PA"].value, X, Y)
        return X, Y/self["q"].value #Axis_Ratio_Cartesian(self["q"].value, X, Y, self["PA"].value, inv_scale = True)
        
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        XX, YY = self.transform_coordinates(X, Y)
        return self.radial_model(self.radius_metric(XX, YY), image)
