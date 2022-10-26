from .model_object import BaseModel
from autoprof.utils.initialize import isophotes
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr
import torch
import numpy as np

class Galaxy_Model(BaseModel):
    """General galaxy model to be subclassed for any specific
    representation. Defines a galaxy as an object with a position
    angle and axis ratio, or effectively a tilted disk. Most
    subclassing models should simply define a radial model or update
    to the coordinate transform.

    """
    model_type = " ".join(("galaxy", BaseModel.model_type))
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0,1), "uncertainty": 0.03},
        "PA": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
    }

    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["PA"].value is not None:
            self["PA"].set_value(self.parameter_specs["PA"]["value"] * np.pi / 180, override_locked = True)

    def initialize(self):
        super().initialize()
        if not (self["PA"].value is None or self["q"].value is None):
            return
        with torch.no_grad():
            target_area = self.target[self.fit_window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            icenter = coord_to_index(self["center"].value[0], self["center"].value[1], target_area)
            if self["PA"].value is None:
                iso_info = isophotes(
                    target_area.data.detach().numpy() - edge_average,
                    (icenter[1].detach().item(), icenter[0].detach().item()),
                    threshold = 3*edge_scatter,
                    pa = 0., q = 1., n_isophotes = 15
                )
                self["PA"].set_value((-Angle_Average(list(iso["phase2"] for iso in iso_info[-int(len(iso_info)/3):]))/2) % np.pi, override_locked = True)
            if self["q"].value is None:
                q_samples = np.linspace(0.1,0.9,15)
                iso_info = isophotes(
                    target_area.data.detach().numpy() - edge_average,
                    (icenter[1].detach().item(), icenter[0].detach().item()),
                    threshold = 3*edge_scatter,
                    pa = self["PA"].value.detach().item(), q = q_samples,
                ) 
                self["q"].set_value(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))], override_locked = True)

    def radius_metric(self, X, Y):
        return torch.sqrt((torch.abs(X)+1e-6)**2 + (torch.abs(Y)+1e-6)**2) # epsilon added for numerical stability of gradient

    def transform_coordinates(self, X, Y):
        return Axis_Ratio_Cartesian(self["q"].value, X, Y, self["PA"].value, inv_scale = True)
        
    def evaluate_model(self, image):
        X, Y = image.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        XX, YY = self.transform_coordinates(X, Y)
        
        return self.radial_model(self.radius_metric(XX, YY), image)
