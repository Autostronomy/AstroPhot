from .sky_model_object import Sky_Model
import numpy as np
from scipy.stats import iqr

class FlatSky(Sky_Model):

    model_type = " ".join(("flat", Sky_Model.model_type))
    parameter_specs = {
        "sky": {"units": "flux"},
        "noise": {"units": "flux", "limits": (0,None)},
    }
    
    def initialize(self, target = None):        
        super().initialize(target)

        if target is None:
            target = self.target
        if self["sky"].value is None:
            self["sky"].set_value(
                np.median(target.get_area(self.model_image.origin, self.window_shape).data),
                override_fixed=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(target.get_area(self.model_image.origin, self.window_shape).data, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0) / np.sqrt(self.window_shape[0] * self.window_shape[1]),
                override_fixed=True,
            )
        if self["noise"].value is None:
            self["noise"].set_value(
                iqr(target.get_area(self.model_image.origin, self.window_shape).data, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0,
                override_fixed=True,
            )
        if self["noise"].uncertainty is None:
            self["noise"].set_uncertainty(
                self["noise"].value / np.sqrt(2 * self.window_shape[0] * self.window_shape[1] - 2),
                override_fixed=True,
            )

    def sample_model(self):
        super().sample_model()

        self.model_image += self["sky"].value
