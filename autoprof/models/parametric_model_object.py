from .model_object import BaseModel
import numpy as np
from autoprof.utils.initialize import isophotes
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr

class Parametric_Model(BaseModel):

    model_type = " ".join(("parametric", BaseModel.model_type))
    parameter_specs = {
        "q": {"units": "b/a", "limits": (0,1), "uncertainty": 0.03},
        "PA": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
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
        
        target_area = target[self.window]
        icenter = coord_to_index(self["center_x"].value, self["center_y"].value, target_area)
        threshold = 3*iqr(target_area.data, rng = (16,84))/2
        if self["PA"].value is None:
            iso_info = isophotes(
                target_area.data,
                (icenter[1], icenter[0]),
                threshold = threshold,
                pa = 0., q = 1., n_isophotes = 3
            )
            self["PA"].set_value((-Angle_Average(list(iso["phase2"] for iso in iso_info))/2) % np.pi, override_fixed = True)
        if self["q"].value is None:
            q_samples = np.linspace(0.1,0.9,15)
            iso_info = isophotes(
                target_area.data,
                (icenter[1], icenter[0]),
                threshold = threshold,
                pa = self["PA"].value, q = q_samples,
            ) 
            self["q"].set_value(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))], override_fixed = True)

    def radial_model(self, R, sample_image):
        raise NotImplementedError("Parametric_Model object doesnt have radial_model function, use a subclass of Parametric_Model")
            
    def sample_model(self, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image

        super().sample_model(sample_image)

        XX, YY = sample_image.get_coordinate_meshgrid(self["center_x"].value, self["center_y"].value)
        XX, YY = Rotate_Cartesian(-self["PA"].value, XX, YY)
        RR = np.sqrt(XX**2 + (YY / self["q"].value)**2)
        
        sample_image += self.radial_model(RR, sample_image)
        
