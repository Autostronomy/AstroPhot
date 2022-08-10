from .model_object import BaseModel
import numpy as np
from autoprof.utils.initialize import isophotes
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr

class Galaxy_Model(BaseModel):

    model_type = " ".join(("galaxy", BaseModel.model_type))
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0,1), "uncertainty": 0.03},
        "PA": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
    }
    parameter_qualities = {
        "q": {"form": "value"},
        "PA": {"form": "value"},
    }

    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["PA"].value is not None:
            self["PA"].set_value(self["PA"].value * np.pi / 180, override_fixed = True)

    def initialize(self, target = None):
        if target is None:
            target = self.target
            
        super().initialize(target)
        if not (self["PA"].value is None or self["q"].value is None):
            return
        print(self.name)
        target_area = target[self.window]
        edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng = (16,84))/2
        icenter = coord_to_index(self["center"][0].value, self["center"][1].value, target_area)
        if self["PA"].value is None:
            iso_info = isophotes(
                target_area.data - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = 0., q = 1., n_isophotes = 15
            )
            self["PA"].set_value((-Angle_Average(list(iso["phase2"] for iso in iso_info[-int(len(iso_info)/3):]))/2) % np.pi, override_fixed = True)
        if self["q"].value is None:
            q_samples = np.linspace(0.1,0.9,15)
            iso_info = isophotes(
                target_area.data - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value, q = q_samples,
            ) 
            self["q"].set_value(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))], override_fixed = True)

    def radius_metric(self, X, Y):
        return np.sqrt(X**2 + Y**2)

    def transform_coordinates(self, X, Y):
        return Axis_Ratio_Cartesian(self["q"].value, X, Y, self["PA"].value, inv_scale = True)
        
    def evaluate_model(self, X, Y, image):
        
        X, Y = self.transform_coordinates(X, Y)
        
        return self.radial_model(self.radius_metric(X, Y), image)
